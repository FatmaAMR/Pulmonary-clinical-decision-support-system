from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from logic import router as classifier_router

app = FastAPI(title="Pediatric Chest X-ray API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classifier_router)


@app.get("/health")
async def health():
    return {"status": "ok"}