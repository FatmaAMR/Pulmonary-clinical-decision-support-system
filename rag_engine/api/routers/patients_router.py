import re
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request, status
from pydantic import BaseModel
from api.dependencies import get_pipeline, get_retrieval_pipeline, get_rag_pipeline
from pipeline.ingestion_pipeline import IngestionPipeline
from pipeline.retrieval_pipeline import RetrievalPipeline
from pipeline.rag_pipeline import EndToEndRAGPipeline
from core.patient_store import PatientRecordStore

router = APIRouter(prefix="/patients", tags=["Patient Management & Clinical Context"])

CONVERSATIONAL_PATTERNS = re.compile(
    r"^(hi|hello|hey|good morning|good evening|thanks|thank you|what can you do|how can you help|who are you|help me|ازيك|اهلا|مرحبا|هتساعدني في ايه|بتعمل ايه|ممكن تساعدني|شكرا)[\s\?!\.,؟]*$",
    re.IGNORECASE,
)


def get_patient_store(request: Request) -> PatientRecordStore:
    return request.app.state.patient_store


class ClinicalChatRequest(BaseModel):
    message: str
    patient_id: Optional[str] = None
    top_k: int = 4
    mode: str = "hybrid"
    classifier_summary: Optional[str] = None


class XrayPredictionUpdate(BaseModel):
    filename: str = "chest_xray.png"
    prediction: str


@router.get("")
def list_all_patients(store: PatientRecordStore = Depends(get_patient_store)):
    return {"patients": store.list_patients()}


@router.get("/{patient_id}")
def get_patient_by_id(patient_id: str, store: PatientRecordStore = Depends(get_patient_store)):
    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient '{patient_id}' not found.")
    return patient


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_or_update_patient(
    patient_id: str = Form(...),
    age: str = Form(""),
    clinical_note: str = Form(""),
    last_xray_prediction: Optional[str] = Form(None),
    history_file: Optional[UploadFile] = File(None),
    xray_file: Optional[UploadFile] = File(None),
    store: PatientRecordStore = Depends(get_patient_store),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
):
    history_file_id = None
    history_filename = None
    parsed_history_text = None

    if history_file and history_file.filename:
        raw_bytes = await history_file.read()
        if raw_bytes:
            history_filename = history_file.filename
            upload_res = pipeline.upload_raw_document(
                filename=history_filename,
                raw_bytes=raw_bytes,
                source_name=f"PatientHistory:{patient_id.strip()}",
                version_date="Clinical-History",
                source_link=f"/documents/{upload_res['file_id']}/view" if False else "db://patient_history",
            )
            history_file_id = upload_res["file_id"]
            chunk_res = pipeline.run_chunking_for_file(history_file_id, force_rechunk=True)
            if chunk_res["chunk_count"] > 0:
                retrieval_pipeline.index_file_chunks(history_file_id)

            parsed_docs = pipeline.store.get_parsed_documents_by_file(history_file_id)
            parsed_history_text = "\n\n".join(d.content for d in parsed_docs).strip()

    xray_filename = xray_file.filename if (xray_file and xray_file.filename) else None

    saved = store.upsert_patient(
        patient_id=patient_id,
        age=age,
        clinical_note=clinical_note,
        history_file_id=history_file_id,
        history_filename=history_filename,
        parsed_history_text=parsed_history_text,
        last_xray_filename=xray_filename,
        last_xray_prediction=last_xray_prediction,
    )
    return {"status": "saved", "patient": saved}


@router.patch("/{patient_id}/xray")
def update_patient_xray_prediction(
    patient_id: str,
    payload: XrayPredictionUpdate,
    store: PatientRecordStore = Depends(get_patient_store),
):
    saved = store.upsert_patient(
        patient_id=patient_id,
        last_xray_filename=payload.filename,
        last_xray_prediction=payload.prediction,
    )
    return {"status": "updated", "patient": saved}


@router.delete("/{patient_id}")
def delete_patient_record(
    patient_id: str,
    store: PatientRecordStore = Depends(get_patient_store),
    pipeline: IngestionPipeline = Depends(get_pipeline),
):
    existing = store.get_patient(patient_id)
    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient '{patient_id}' not found.")

    if existing.get("history_file_id"):
        try:
            pipeline.store.delete_file(existing["history_file_id"])
        except Exception:
            pass

    store.delete_patient(patient_id)
    return {"status": "deleted", "patient_id": patient_id}


@router.post("/clinical-chat")
def clinical_chat_with_patient_context(
    req: ClinicalChatRequest,
    store: PatientRecordStore = Depends(get_patient_store),
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    patient_record = None
    patient_context_block = ""

    if req.patient_id and req.patient_id.strip():
        patient_record = store.get_patient(req.patient_id.strip())
        if patient_record:
            parts = [f"Patient ID: {patient_record['patient_id']}"]
            if patient_record.get("age"):
                parts.append(f"Age: {patient_record['age']}")
            if patient_record.get("clinical_note"):
                parts.append(f"Clinician Notes: {patient_record['clinical_note']}")
            if patient_record.get("last_xray_prediction"):
                parts.append(f"Recorded Chest X-Ray Classifier Output: {patient_record['last_xray_prediction']}")
            if patient_record.get("parsed_history_text"):
                parts.append(f"Parsed Medical History Document:\n{patient_record['parsed_history_text'][:2500]}")
            patient_context_block = "\n".join(parts)

    if req.classifier_summary:
        patient_context_block = (
            f"{patient_context_block}\nLatest Attached X-Ray Classifier Result: {req.classifier_summary}".strip()
        )

    clean_msg = req.message.strip()

    # Handle general greetings / capability questions directly without fake citations
    if CONVERSATIONAL_PATTERNS.match(clean_msg) or len(clean_msg) < 4:
        patient_note = ""
        if patient_record:
            patient_note = (
                f"\n\nI currently have Patient {patient_record['patient_id']} loaded "
                f"(Age: {patient_record.get('age') or 'N/A'}, "
                f"X-Ray Result: {patient_record.get('last_xray_prediction') or 'Not analyzed yet'}). "
                "You can ask me to correlate their history and X-ray findings with pediatric pneumonia guidelines."
            )
        return {
            "query": req.message,
            "answer": (
                "Hello! I am CHESTWISE, your pediatric pulmonary clinical decision support assistant. "
                "I can help you:\n"
                "• Analyze pediatric chest X-rays using the connected Pneumonia Classifier.\n"
                "• Retrieve evidence-based pediatric pneumonia guidelines with clickable source links.\n"
                "• Review a patient's parsed medical history and correlate it with clinical protocols."
                f"{patient_note}"
            ),
            "sources_cited": [],
            "abstained": False,
            "model_name": "conversational-guard",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "patient_context_used": bool(patient_record),
            "patient_summary": {
                "patient_id": patient_record["patient_id"],
                "age": patient_record.get("age", ""),
                "history_filename": patient_record.get("history_filename", ""),
                "last_xray_prediction": patient_record.get("last_xray_prediction", ""),
            } if patient_record else None,
        }

    augmented_query = clean_msg
    if patient_context_block:
        augmented_query = (
            f"Patient Context:\n{patient_context_block}\n\n"
            f"Clinician Question: {clean_msg}"
        )

    result = rag_pipeline.run(
        query=augmented_query,
        top_k=req.top_k,
        mode=req.mode,
    )

    return {
        "query": req.message,
        "answer": result.answer,
        "sources_cited": result.sources_cited,
        "abstained": result.abstained,
        "model_name": result.model_name,
        "token_usage": result.token_usage,
        "patient_context_used": bool(patient_context_block),
        "patient_summary": {
            "patient_id": patient_record["patient_id"],
            "age": patient_record.get("age", ""),
            "history_filename": patient_record.get("history_filename", ""),
            "last_xray_prediction": patient_record.get("last_xray_prediction", ""),
        } if patient_record else None,
    }
