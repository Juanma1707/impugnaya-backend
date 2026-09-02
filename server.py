import os
import json
import time
import httpx
from datetime import datetime
from typing import Dict, Any, Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ImpugnaYa API - Motor de Prescripción de Tránsito Perú")

# Configuración CORS abierta
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CULQI_SECRET_KEY = os.environ.get("CULQI_SECRET_KEY", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# --- MODELOS DE DATOS ---
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

class CulqiCardChargeRequest(BaseModel):
    token_id: str
    amount: int = 990
    email: str
    concept: Optional[str] = "Expediente Legal ImpugnaYa"

# --- MOTOR DE CÁLCULO Y FUNDAMENTACIÓN LEGAL (TUO LEY 27444 Y LEY 26979) ---
def evaluate_traffic_ticket(data: InfractionEvaluationRequest) -> Dict[str, Any]:
    current_year = 2026
    diff_years = max(current_year - data.year, 0)

    # Valores oficiales aproximados según código de infracción del MTC
    code_upper = data.code.upper().strip()
    if code_upper.startswith("M01") or code_upper.startswith("M02"):
        fine_amount = 2575.00  # 50% UIT
        severity = "Muy Grave (Riesgo de Cancelación/Retención)"
    elif code_upper.startswith("M20") or code_upper.startswith("M"):
        fine_amount = 927.00   # 18% UIT
        severity = "Muy Grave"
    elif code_upper.startswith("G"):
        fine_amount = 412.00   # 8% UIT
        severity = "Grave"
    else:
        fine_amount = 206.00   # 4% UIT (Leve)

    # Gastos de cobranza coactiva y costas accesorias estándar (estimado 15%)
    costas_coactivas = round(fine_amount * 0.15, 2)
    ahorro_total = fine_amount + costas_coactivas

    # Análisis jurídico estricto de procedencia
    if diff_years >= 4:
        prescription_status = "PROCEDENTE_TOTAL"
        diagnosis_title = "Prescripción Definitiva Consumada"
        legal_basis = (
            f"Han transcurrido {diff_years} años desde la emisión de la papeleta en el año {data.year}. "
            "Conforme al Artículo 252.1 del TUO de la Ley N° 27444 (D.S. 004-2019-JUS) y Art. 23 de la Ley N° 26979, "
            "ha operado la PRESCRIPCIÓN EXTINTIVA TOTAL de la exigibilidad de la sanción y costas. "
            "La administración está legalmente obligada a declarar el archivo definitivo y levantar de oficio o a instancia de parte cualquier orden de captura."
        )
    elif diff_years >= 2:
        prescription_status = "PROCEDENTE_COACTIVA"
        diagnosis_title = "Prescripción de Ejecución y Vicio de Notificación"
        legal_basis = (
            f"Con {diff_years} años de antigüedad, la facultad sancionadora ordinaria ha vencido. "
            "Para que la entidad pueda cobrar coactivamente, debió acreditar notificación válida de la Resolución de Sanción "
            "dentro de los 2 años. Al no existir notificación válida bajo puerta conforme al Art. 21 de la Ley 27444, "
            "procede la suspensión de la ejecución coactiva (Art. 16.1.e Ley 26979) y desafectación de captura."
        )
    else:
        prescription_status = "NULIDAD_NOTIFICACION"
        diagnosis_title = "Nulidad por Vicio Formal de Notificación"
        legal_basis = (
            "Infracción dentro del periodo de cobro ordinario. Procede la solicitud de Nulidad de Oficio "
            "por vicio insanable de notificación defectuosa y vulneración del derecho constitucional al debido procedimiento (D.S. 004-2020-MTC)."
        )

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

# --- ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "ok", "service": "ImpugnaYa API Online - Tránsito Perú"}

@app.post("/api/evaluate")
async def evaluate_endpoint(req: InfractionEvaluationRequest):
    return evaluate_traffic_ticket(req)

@app.post("/api/ai-consult")
async def ai_consult_endpoint(req: AIConsultRequest):
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "Falta configurar GEMINI_API_KEY en Render."}

    system_instruction = (
        "Eres el Asesor Legal de ImpugnaYa.pe, especialista en derecho administrativo de tránsito en el Perú "
        "(TUO Ley 27444 - D.S. 004-2019-JUS, Ley de Ejecución Coactiva 26979, D.S. 016-2009-MTC y fiscalización de SAT Lima, Callao y SUTRAN).\n"
        "Tu misión es responder con solvencia, empatía y base legal clara a los conductores y dueños de vehículos:\n"
        "1. Explica que las multas de tránsito prescriben a los 2 o 4 años conforme al Art. 252 de la Ley 27444 si no fueron cobradas legítimamente.\n"
        "2. Detalla que una orden de captura vehicular por una papeleta prescrita es ILEGAL (vía de hecho sancionable según Art. 16 Ley 26979).\n"
        "3. Sé conciso y formal pero accesible para el ciudadano peruano. Invítalo a generar su escrito formal de 2 páginas para mesa de partes virtual."
    )

    formatted_contents = []
    for msg in req.messages:
        formatted_contents.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [{"text": msg.text}]
        })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": formatted_contents,
        "generationConfig": {"temperature": 0.2}
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            res = await client.post(url, json=payload)
            data = res.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "ok", "reply": reply}
        except Exception as e:
            return {"status": "error", "message": f"Error Gemini: {str(e)}"}

# --- ENDPOINT CULQI: CREACIÓN DE ORDEN OFICIAL PARA YAPE ---
@app.post("/api/create-order")
async def create_culqi_order(req: CreateOrderRequest):
    if not CULQI_SECRET_KEY:
        return {"status": "error", "message": "Falta CULQI_SECRET_KEY en el servidor."}

    expiration = int(time.time()) + (2 * 3600)  # Vigencia de 2 horas
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
                return {
                    "status": "ok",
                    "order_id": data["id"],
                    "order_number": order_number
                }
            else:
                user_msg = data.get("user_message") or data.get("merchant_message") or "No se pudo generar la orden."
                return {"status": "error", "message": user_msg}
        except Exception as e:
            return {"status": "error", "message": f"Error conectando con Culqi: {str(e)}"}

# --- ENDPOINT PARA COBROS CON TARJETA ---
@app.post("/api/culqi-charge")
async def process_card_charge(req: CulqiCardChargeRequest):
    if not CULQI_SECRET_KEY:
        return {"status": "error", "message": "Falta CULQI_SECRET_KEY en el servidor."}

    charge_payload = {
        "amount": req.amount,
        "currency_code": "PEN",
        "email": req.email,
        "source_id": req.token_id,
        "description": req.concept
    }

    headers = {
        "Authorization": f"Bearer {CULQI_SECRET_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            res = await client.post("https://api.culqi.com/v2/charges", json=charge_payload, headers=headers)
            data = res.json()
            if res.status_code in [200, 201] and data.get("object") == "charge":
                return {
                    "status": "approved",
                    "charge_id": data.get("id"),
                    "amount": data.get("amount") / 100.0,
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M")
                }
            else:
                user_msg = data.get("user_message") or data.get("merchant_message") or "Pago no aprobado."
                return {"status": "rejected", "message": user_msg}
        except Exception as e:
            return {"status": "error", "message": f"Error pasarela: {str(e)}"}
