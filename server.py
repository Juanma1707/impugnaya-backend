import os
import json
import time
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ImpugnaYa API - Prescripción de Tránsito Perú")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CULQI_SECRET_KEY = os.environ.get("CULQI_SECRET_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# --- MODELOS ---
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

class CreateOrderRequest(BaseModel):
    amount: int = 990
    currency: str = "PEN"
    description: str = "Expediente Legal de Prescripcion y Levantamiento de Captura"
    email: Optional[str] = "cliente@impugnaya.pe"

# --- EVALUACIÓN DE PAPELETA ---
def evaluate_traffic_ticket(data: InfractionEvaluationRequest) -> Dict[str, Any]:
    current_year = 2026
    diff_years = max(current_year - data.year, 0)

    code_upper = data.code.upper().strip()
    if code_upper.startswith("M01") or code_upper.startswith("M02"):
        fine_amount = 2575.00
        severity = "Muy Grave (Riesgo Crítico)"
    elif code_upper.startswith("M20") or code_upper.startswith("M"):
        fine_amount = 927.00
        severity = "Muy Grave"
    elif code_upper.startswith("G"):
        fine_amount = 412.00
        severity = "Grave"
    else:
        fine_amount = 206.00
        severity = "Leve"

    costas_coactivas = round(fine_amount * 0.15, 2)
    ahorro_total = fine_amount + costas_coactivas

    if diff_years >= 4:
        prescription_status = "PROCEDENTE_TOTAL"
        diagnosis_title = "Prescripción Definitiva Consumada"
        legal_basis = (
            f"Han transcurrido {diff_years} años desde la infracción ({data.year}). "
            "Conforme al Art. 252.1 del TUO de la Ley N° 27444 y Ley N° 26979, ha operado la "
            "PRESCRIPCIÓN EXTINTIVA TOTAL. La autoridad está obligada a archivar y levantar la captura."
        )
    elif diff_years >= 2:
        prescription_status = "PROCEDENTE_COACTIVA"
        diagnosis_title = "Prescripción de Ejecución y Vicio Formal"
        legal_basis = (
            f"Con {diff_years} años transcurridos, la potestad sancionadora ordinaria ha vencido. "
            "Procede la suspensión coactiva según Art. 16.1.e de la Ley 26979."
        )
    else:
        prescription_status = "NULIDAD_NOTIFICACION"
        diagnosis_title = "Nulidad por Vicio de Notificación"
        legal_basis = "Infracción reciente. Procede solicitud de nulidad por falta de notificación reglamentaria bajo puerta."

    return {
        "administrado": {
            "nombre": data.driver_name.upper(),
            "dni": data.driver_dni,
            "domicilio": data.driver_address.upper(),
            "email": data.driver_email,
            "telefono": data.phone or "NO ESPECIFICADO"
        },
        "infraccion": {
            "placa": data.plate.upper(),
            "papeleta": data.ticket_number.upper(),
            "codigo": code_upper,
            "entidad": data.authority.upper(),
            "anio": data.year,
            "gravedad": severity,
            "multa_nominal": fine_amount,
            "costas_coactivas": costas_coactivas,
            "ahorro_total": ahorro_total
        },
        "dictamen": {
            "estado": prescription_status,
            "titulo": diagnosis_title,
            "antiguedad_anios": diff_years,
            "fundamentacion": legal_basis
        }
    }

# --- RUTAS ---
@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ImpugnaYa API Activo",
        "has_gemini": bool(GEMINI_API_KEY),
        "has_culqi": bool(CULQI_SECRET_KEY)
    }

@app.post("/api/evaluate")
async def evaluate_endpoint(req: InfractionEvaluationRequest):
    return evaluate_traffic_ticket(req)

@app.post("/api/ai-consult")
async def ai_consult_endpoint(req: AIConsultRequest):
    if not GEMINI_API_KEY:
        return {"status": "error", "reply": "Falta configurar GEMINI_API_KEY en Render."}

    system_instruction = (
        "Eres el Asesor Legal de Tránsito de ImpugnaYa.pe, especialista en derecho administrativo peruano "
        "(TUO Ley 27444, D.S. 016-2009-MTC, Ley de Ejecución Coactiva 26979, SAT Lima, Callao y SATP Piura).\n"
        "Reglas:\n"
        "1. Responde de forma personalizada a lo que el conductor pregunta.\n"
        "2. Si la papeleta tiene más de 2 o 4 años, aclara que prescribió según el Art. 252 del TUO de la Ley 27444.\n"
        "3. Mantén un tono formal, profesional y empático en 2 párrafos concisos."
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

    # Modelos oficiales vigentes
    candidate_models = ["gemini-1.5-flash", "gemini-2.0-flash"]
    last_error = ""

    async with httpx.AsyncClient(timeout=25.0) as client:
        for model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                res = await client.post(url, json=payload, headers=headers)
                print(f"[{model}] Status: {res.status_code}", flush=True)
                
                if res.status_code == 200:
                    data = res.json()
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"status": "ok", "reply": reply}
                else:
                    last_error = f"HTTP {res.status_code}: {res.text}"
                    print(f"[{model}] Error: {last_error}", flush=True)
            except Exception as e:
                last_error = str(e)
                print(f"[{model}] Excepcion: {last_error}", flush=True)

    return {"status": "error", "reply": f"Respuesta de Google AI: {last_error}"}

@app.post("/api/create-order")
async def create_culqi_order(req: CreateOrderRequest):
    if not CULQI_SECRET_KEY:
        return {"status": "error", "message": "Falta CULQI_SECRET_KEY en Render."}

    expiration = int(time.time()) + (2 * 3600)
    order_number = f"IMP-{int(time.time())}"

    order_payload = {
        "amount": req.amount,
        "currency_code": req.currency,
        "description": req.description,
        "order_number": order_number,
        "client_details": {
            "first_name": "Conductor",
            "last_name": "ImpugnaYa",
            "email": req.email,
            "phone_number": "957418893"
        },
        "expiration_date": expiration
    }

    headers = {
        "Authorization": f"Bearer {CULQI_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            res = await client.post("https://api.culqi.com/v2/orders", json=order_payload, headers=headers)
            data = res.json()
            if res.status_code in [200, 201] and "id" in data:
                return {"status": "ok", "order_id": data["id"], "order_number": order_number}
            else:
                return {"status": "error", "message": data.get("user_message", "Error al crear orden")}
        except Exception as e:
            return {"status": "error", "message": str(e)}
