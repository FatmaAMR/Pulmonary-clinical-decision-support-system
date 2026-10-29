import re
import base64
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request, Query, status
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
    clinical_note: Optional[str] = None
    image_data_url: Optional[str] = None


class PatientProfilePatch(BaseModel):
    patient_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[str] = None
    weight_kg: Optional[str] = None
    clinical_note: Optional[str] = None
    proposed_treatment: Optional[str] = None
    manual_history_text: Optional[str] = None


class TreatmentEvaluationRequest(BaseModel):
    proposed_treatment: str
    weight_kg: Optional[str] = None
    age: Optional[str] = None
    clinical_note: Optional[str] = None
    top_k: int = 4


def _build_patient_context_string(patient_record: dict) -> str:
    parts = [f"Patient ID: {patient_record['patient_id']}"]
    if patient_record.get("patient_name"):
        parts.append(f"Patient Name: {patient_record['patient_name']}")
    if patient_record.get("gender"):
        parts.append(f"Gender: {patient_record['gender']}")
    if patient_record.get("age"):
        parts.append(f"Age: {patient_record['age']}")
    if patient_record.get("weight_kg"):
        parts.append(f"Weight: {patient_record['weight_kg']} kg")
    if patient_record.get("clinical_note"):
        parts.append(f"Current Clinical Notes & Vitals: {patient_record['clinical_note']}")
    if patient_record.get("last_xray_prediction"):
        parts.append(f"Latest Chest X-Ray Classifier Output: {patient_record['last_xray_prediction']}")
    else:
        parts.append("Chest X-Ray Status: No chest X-ray uploaded yet.")
    if patient_record.get("proposed_treatment"):
        parts.append(f"Proposed Treatment Regimen: {patient_record['proposed_treatment']}")

    timeline = patient_record.get("timeline") or []
    if len(timeline) > 1:
        history_steps = [
            f"- [{t.get('timestamp', '')}] X-Ray: {t.get('prediction', 'N/A')} ({t.get('filename', '')}) | Note: {t.get('note', 'N/A')}"
            for t in timeline[:5]
        ]
        parts.append("Longitudinal Imaging & Clinical Timeline (Newest first):\n" + "\n".join(history_steps))

    if patient_record.get("parsed_history_text"):
        parts.append(f"Patient Medical History:\n{patient_record['parsed_history_text'][:2500]}")
    return "\n".join(parts)


@router.get("/next-id")
def get_next_generated_patient_id(store: PatientRecordStore = Depends(get_patient_store)):
    return {"next_patient_id": store.generate_next_patient_id()}


@router.get("")
def list_or_search_patients(
    q: Optional[str] = Query(default="", description="Search by Patient ID or Name"),
    store: PatientRecordStore = Depends(get_patient_store),
):
    return {
        "patients": store.search_patients(q or ""),
        "next_patient_id": store.generate_next_patient_id(),
    }


@router.get("/{patient_id}")
def get_patient_by_id(patient_id: str, store: PatientRecordStore = Depends(get_patient_store)):
    patient = store.get_patient(patient_id)
    if not patient:
        matches = store.search_patients(patient_id)
        if matches:
            return matches[0]
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Patient '{patient_id}' not found.")
    return patient


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_or_update_patient(
    patient_id: str = Form("AUTO"),
    patient_name: str = Form(""),
    gender: str = Form(""),
    age: str = Form(""),
    weight_kg: str = Form(""),
    clinical_note: str = Form(""),
    manual_history_text: str = Form(""),
    proposed_treatment: str = Form(""),
    evaluate_treatment_now: bool = Form(False),
    last_xray_prediction: Optional[str] = Form(None),
    history_file: Optional[UploadFile] = File(None),
    xray_file: Optional[UploadFile] = File(None),
    store: PatientRecordStore = Depends(get_patient_store),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    resolved_id = (patient_id or "").strip()
    if not resolved_id or resolved_id.upper() == "AUTO":
        resolved_id = store.generate_next_patient_id()

    history_file_id = None
    history_filename = None
    parsed_history_text = None
    rag_chunks_indexed = 0

    file_extracted_text = ""
    if history_file and history_file.filename:
        raw_bytes = await history_file.read()
        if raw_bytes:
            history_filename = history_file.filename
            upload_res = pipeline.upload_raw_document(
                filename=history_filename,
                raw_bytes=raw_bytes,
                source_name=f"PatientHistory:{resolved_id}",
                version_date="Clinical-History",
                source_link="db://patient_history",
            )
            history_file_id = upload_res["file_id"]
            chunk_res = pipeline.run_chunking_for_file(history_file_id, force_rechunk=True)
            rag_chunks_indexed = chunk_res.get("chunk_count", 0)
            if rag_chunks_indexed > 0:
                retrieval_pipeline.index_file_chunks(history_file_id)

            parsed_docs = pipeline.store.get_parsed_documents_by_file(history_file_id)
            file_extracted_text = "\n\n".join(d.content for d in parsed_docs).strip()

    manual_clean = manual_history_text.strip()
    if manual_clean and file_extracted_text:
        parsed_history_text = f"{manual_clean}\n\n--- Attached History Document ({history_filename}) ---\n{file_extracted_text}"
    elif manual_clean:
        parsed_history_text = manual_clean
        history_filename = history_filename or "Written Clinical History"
        upload_res = pipeline.upload_raw_document(
            filename=f"{resolved_id}_history.txt",
            raw_bytes=manual_clean.encode("utf-8"),
            source_name=f"PatientHistory:{resolved_id}",
            version_date="Clinical-History",
            source_link="db://patient_history",
        )
        history_file_id = upload_res["file_id"]
        chunk_res = pipeline.run_chunking_for_file(history_file_id, force_rechunk=True)
        rag_chunks_indexed = chunk_res.get("chunk_count", 0)
        if rag_chunks_indexed > 0:
            retrieval_pipeline.index_file_chunks(history_file_id)
    elif file_extracted_text:
        parsed_history_text = file_extracted_text

    xray_filename = None
    timeline_entry = None
    clean_pred = (last_xray_prediction or "").strip()
    if "404" in clean_pred or clean_pred.lower().startswith("pending"):
        clean_pred = ""

    if xray_file and xray_file.filename:
        xray_filename = xray_file.filename
        xray_bytes = await xray_file.read()
        ext = xray_filename.lower().split(".")[-1]
        mime = "image/png" if ext == "png" else "image/jpeg"
        b64_img = f"data:{mime};base64," + base64.b64encode(xray_bytes).decode("ascii") if len(xray_bytes) < 1_500_000 else ""
        timeline_entry = {
            "type": "Imaging & Clinical Visit",
            "filename": xray_filename,
            "prediction": clean_pred or "Uploaded X-Ray",
            "note": clinical_note or "",
            "treatment": proposed_treatment or "",
            "image_data_url": b64_img,
        }

    saved = store.upsert_patient(
        patient_id=resolved_id,
        patient_name=patient_name,
        gender=gender,
        age=age,
        weight_kg=weight_kg,
        clinical_note=clinical_note,
        proposed_treatment=proposed_treatment,
        history_file_id=history_file_id,
        history_filename=history_filename,
        parsed_history_text=parsed_history_text,
        last_xray_filename=xray_filename,
        last_xray_prediction=clean_pred if clean_pred else None,
        new_timeline_entry=timeline_entry,
    )

    evaluation_sources = []
    if evaluate_treatment_now and proposed_treatment.strip():
        eval_res = _run_treatment_evaluation(saved, rag_pipeline, top_k=4)
        saved = store.upsert_patient(
            patient_id=resolved_id,
            ai_treatment_evaluation=eval_res["answer"],
        )
        evaluation_sources = eval_res["sources_cited"]

    return {
        "status": "saved",
        "rag_chunks_indexed": rag_chunks_indexed,
        "patient": saved,
        "evaluation_sources": evaluation_sources,
    }


@router.patch("/{patient_id}/profile")
def update_patient_profile_fields(
    patient_id: str,
    payload: PatientProfilePatch,
    store: PatientRecordStore = Depends(get_patient_store),
):
    saved = store.upsert_patient(
        patient_id=patient_id,
        patient_name=payload.patient_name,
        gender=payload.gender,
        age=payload.age,
        weight_kg=payload.weight_kg,
        clinical_note=payload.clinical_note,
        proposed_treatment=payload.proposed_treatment,
        parsed_history_text=payload.manual_history_text,
    )
    return {"status": "updated", "patient": saved}


def _run_treatment_evaluation(
    patient_record: dict,
    rag_pipeline: EndToEndRAGPipeline,
    top_k: int = 4,
) -> dict:
    profile_text = _build_patient_context_string(patient_record)
    eval_prompt = (
        f"PATIENT CLINICAL PROFILE:\n{profile_text}\n\n"
        f"PROPOSED TREATMENT PLAN:\n{patient_record.get('proposed_treatment', '')}\n\n"
        "Evaluate the proposed medication and treatment regimen against the pediatric pneumonia clinical guidelines:\n"
        "1. Appropriateness: Does the drug choice and route match the patient's severity, age, clinical notes, and Chest X-ray findings?\n"
        "2. Dosage Check: Verify the mg/kg/day calculation against guideline recommendations if weight is provided.\n"
        "3. Clinical Recommendations: Highlight any red flags or follow-up considerations with inline [Source X] citations."
    )
    res = rag_pipeline.run(query=eval_prompt, top_k=top_k, mode="hybrid")
    return {
        "answer": res.answer,
        "sources_cited": res.sources_cited,
        "model_name": res.model_name,
    }


@router.post("/{patient_id}/evaluate-treatment")
def evaluate_patient_treatment_plan(
    patient_id: str,
    req: TreatmentEvaluationRequest,
    store: PatientRecordStore = Depends(get_patient_store),
    rag_pipeline: EndToEndRAGPipeline = Depends(get_rag_pipeline),
):
    existing = store.get_patient(patient_id)
    if not existing:
        existing = store.upsert_patient(
            patient_id=patient_id,
            age=req.age or "",
            weight_kg=req.weight_kg or "",
            clinical_note=req.clinical_note or "",
            proposed_treatment=req.proposed_treatment,
        )
    else:
        existing = store.upsert_patient(
            patient_id=patient_id,
            age=req.age if req.age is not None else existing.get("age", ""),
            weight_kg=req.weight_kg if req.weight_kg is not None else existing.get("weight_kg", ""),
            clinical_note=req.clinical_note if req.clinical_note is not None else existing.get("clinical_note", ""),
            proposed_treatment=req.proposed_treatment,
        )

    eval_res = _run_treatment_evaluation(existing, rag_pipeline, top_k=req.top_k)
    updated_patient = store.upsert_patient(
        patient_id=patient_id,
        ai_treatment_evaluation=eval_res["answer"],
    )

    return {
        "status": "evaluated",
        "patient": updated_patient,
        "evaluation": eval_res["answer"],
        "sources_cited": eval_res["sources_cited"],
        "model_name": eval_res["model_name"],
    }


@router.patch("/{patient_id}/xray")
def update_patient_xray_prediction(
    patient_id: str,
    payload: XrayPredictionUpdate,
    store: PatientRecordStore = Depends(get_patient_store),
):
    timeline_entry = {
        "type": "Chest X-Ray Analysis",
        "filename": payload.filename,
        "prediction": payload.prediction,
        "note": payload.clinical_note or "Analyzed via Clinical Workspace",
        "image_data_url": payload.image_data_url or "",
    }
    saved = store.upsert_patient(
        patient_id=patient_id,
        last_xray_filename=payload.filename,
        last_xray_prediction=payload.prediction,
        new_timeline_entry=timeline_entry,
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
        if not patient_record:
            matches = store.search_patients(req.patient_id.strip())
            if matches:
                patient_record = matches[0]
        if patient_record:
            patient_context_block = _build_patient_context_string(patient_record)

    if req.classifier_summary:
        patient_context_block = (
            f"{patient_context_block}\nLatest Attached X-Ray Classifier Result: {req.classifier_summary}".strip()
        )

    clean_msg = req.message.strip()

    if CONVERSATIONAL_PATTERNS.match(clean_msg) or len(clean_msg) < 4:
        patient_note = ""
        if patient_record:
            pname = f" ({patient_record['patient_name']})" if patient_record.get("patient_name") else ""
            xray_state = patient_record.get("last_xray_prediction") or "No X-ray uploaded yet"
            patient_note = (
                f"\n\nActive Patient {patient_record['patient_id']}{pname} is loaded "
                f"(Age: {patient_record.get('age') or 'N/A'}, Weight: {patient_record.get('weight_kg') or 'N/A'} kg, "
                f"X-Ray: {xray_state}). "
                "You can ask me about their clinical findings, X-ray progression, or treatment plan."
            )
        return {
            "query": req.message,
            "answer": (
                "Hello Doctor! I am ChestWise, your pediatric pulmonary clinical assistant. "
                "I am connected to the Guideline Expert and Chest X-Ray Classifier to help you:\n"
                "• Review chest X-ray findings and track imaging progression over time.\n"
                "• Verify antibiotic regimens and weight-based pediatric dosages.\n"
                "• Correlate patient history with clinical guidelines."
                f"{patient_note}"
            ),
            "sources_cited": [],
            "abstained": False,
            "model_name": "conversational-guard",
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "patient_context_used": bool(patient_record),
            "patient_summary": {
                "patient_id": patient_record["patient_id"],
                "patient_name": patient_record.get("patient_name", ""),
                "gender": patient_record.get("gender", ""),
                "age": patient_record.get("age", ""),
                "weight_kg": patient_record.get("weight_kg", ""),
                "proposed_treatment": patient_record.get("proposed_treatment", ""),
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
        "patient_context_used": bool(patient_record),
        "patient_summary": {
            "patient_id": patient_record["patient_id"],
            "patient_name": patient_record.get("patient_name", ""),
            "gender": patient_record.get("gender", ""),
            "age": patient_record.get("age", ""),
            "weight_kg": patient_record.get("weight_kg", ""),
            "proposed_treatment": patient_record.get("proposed_treatment", ""),
            "history_filename": patient_record.get("history_filename", ""),
            "last_xray_prediction": patient_record.get("last_xray_prediction", ""),
        } if patient_record else None,
    }
