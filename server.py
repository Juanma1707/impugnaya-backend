import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="ImpugnaYa Backend API")

# Habilitar CORS para Vercel y local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CULQI_SECRET_KEY = os.getenv("CULQI_SECRET_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class MessageItem(BaseModel):
    role: str
    text: str

class AIConsultRequest(BaseModel):
    messages: list[MessageItem]

class YapeChargeRequest(BaseModel):
    token: str
    amount: int  # en céntimos (ej. 990 para S/ 9.90)
    email: str

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "ImpugnaYa API",
        "has_gemini": bool(GEMINI_API_KEY),
        "has_culqi": bool(CULQI_SECRET_KEY)
    }

@app.post("/api/ai-consult")
async def ai_consult(req: AIConsultRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render")

    system_prompt = (
        "Eres el Asesor Legal de Tránsito de ImpugnaYa.pe, experto en derecho administrativo sancionador en Perú. "
        "Conoces a profundidad el TUO de la Ley N° 27444 (Ley del Procedimiento Administrativo General), la Ley N° 26979 "
        "(Ley de Procedimiento de Ejecución Coactiva), el Texto Único Ordenado del Reglamento Nacional de Tránsito (D.S. 016-2009-MTC) "
        "y los procedimientos del SAT Lima, SATP Piura, SUTRAN y municipios provinciales.\n"
        "Reglas para tus respuestas:\n"
        "1. Si el usuario te cuenta sobre una papeleta de 2020, 2021, 2022 o anterior (más de 2 a 4 años), explícale con calma "
        "que la acción de cobro y exigibilidad ha PRESCRITO conforme al Art. 252 del TUO de la Ley 27444. "
        "Dile que NO debe pagarla, sino ingresar la solicitud formal de prescripción y levantamiento de captura.\n"
        "2. Si la papeleta es reciente (2024-2026), analiza posibles causales de nulidad como defectos de notificación bajo puerta "
        "sin foto reglamentaria o falta de requisitos esenciales en la papeleta.\n"
        "3. Sé conciso, empático, claro y muy seguro de tus fundamentos legales peruanos (máximo 2 a 3 párrafos cortos).\n"
        "4. Al final, indícale amablemente que puede generar el escrito legal formal de 2 páginas con cargo listo para mesa de partes aquí en la web."
    )

    contents = []
    for msg in req.messages:
        role_mapped = "user" if msg.role == "user" else "model"
        contents.append({
            "role": role_mapped,
            "parts": [{"text": msg.text}]
        })

    # Usamos gemini-2.5-flash
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 600
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code != 200:
                # Si falló, probamos fallback automático a gemini-2.0-flash
                fallback_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
                response = await client.post(fallback_url, json=payload)
                if response.status_code != 200:
                    raise HTTPException(status_code=502, detail=f"Error en Gemini API: {response.text}")

            data = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"status": "ok", "reply": reply}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Tiempo de espera agotado con la IA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/charge-yape")
async def charge_yape(req: YapeChargeRequest):
    if not CULQI_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CULQI_SECRET_KEY no configurada")

    url = "https://api.culqi.com/v2/charges"
    headers = {
        "Authorization": f"Bearer {CULQI_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "amount": req.amount,
        "currency_code": "PEN",
        "email": req.email,
        "source_id": req.token,
        "description": "Escrito Oficial ImpugnaYa.pe"
    }

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(url, json=payload, headers=headers)
            data = res.json()
            if res.status_code in [200, 201]:
                return {"status": "ok", "charge": data}
            else:
                return {"status": "error", "message": data.get("user_message", "Error al procesar con Culqi")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
