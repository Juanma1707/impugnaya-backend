import os
import json
import time
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ImpugnaYa API - Prescripción de Tránsito")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CULQI_SECRET_KEY = os.environ.get("CULQI_SECRET_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

class InfractionEvaluationRequest(BaseModel):
    plate: str
    driver_name: str
    driver_dni: str
    driver_address: str
    driver_email: str
    ticket_number: str
    code: str
    authority: str
    year: int
    procedure_type: str
    phone: Optional[str] = ""

class ChatMessage(BaseModel):
    role: str
    text: str

class AIConsultRequest(BaseModel):
    messages: List[ChatMessage]

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "service": "ImpugnaYa API Activo"}

@app.post("/api/ai-consult")
async def ai_consult_endpoint(req: AIConsultRequest):
    if not GEMINI_API_KEY:
        return {"status": "error", "reply": "Falta configurar GEMINI_API_KEY en Render."}

    system_instruction = (
        "Eres el Asesor Legal de Tránsito de ImpugnaYa.pe, especialista en derecho administrativo peruano "
        "(TUO Ley 27444, D.S. 016-2009-MTC, Ley de Ejecución Coactiva 26979, SAT Lima y SATP Piura).\n"
        "Reglas:\n"
        "1. Responde de manera personalizada a lo que el conductor pregunta.\n"
        "2. Si la papeleta tiene más de 2 o 4 años, confirma que prescribió según el Art. 252 del TUO de la Ley 27444.\n"
        "3. Mantén un tono formal, claro y empático en 2 párrafos concisos."
    )

    formatted_contents = []
    for msg in req.messages:
        role = "user" if msg.role == "user" else "model"
        formatted_contents.append({
            "role": role,
            "parts": [{"text": msg.text}]
        })

    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": formatted_contents,
        "generationConfig": {"temperature": 0.3}
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    # Modelo oficial indicado por la API de Google
    models_to_try = ["gemini-3.6-flash", "gemini-2.5-flash"]
    last_error = ""

    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                res = await client.post(url, json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"status": "ok", "reply": reply}
                else:
                    last_error = f"HTTP {res.status_code}: {res.text}"
            except Exception as e:
                last_error = str(e)

    return {"status": "error", "reply": f"Respuesta de Google AI: {last_error}"}
