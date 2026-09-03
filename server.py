@app.post("/api/ai-consult")
async def ai_consult_endpoint(req: AIConsultRequest):
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "No se encontró GEMINI_API_KEY configurada en Render."}

    system_instruction = (
        "Eres el Asesor Legal de Tránsito de ImpugnaYa.pe, especialista en derecho administrativo sancionador peruano "
        "(TUO de la Ley N° 27444, D.S. 016-2009-MTC, Ley de Procedimiento de Ejecución Coactiva 26979, y directivas del SAT y SATP Piura).\n"
        "Reglas:\n"
        "- Si la papeleta es del 2022 o anterior (más de 2 a 4 años), confirma que la acción de cobro prescribió (Art. 252 Ley 27444).\n"
        "- Responde en 2 párrafos concisos, claros y con seguridad jurídica peruana.\n"
        "- Indica que pueden descargar el escrito formal de 2 páginas con cargo aquí en la web."
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

    # Intentamos primero con gemini-1.5-flash y luego con gemini-2.0-flash
    models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        last_error = ""
        for model in models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                res = await client.post(url, json=payload, headers=headers)
                data = res.json()
                
                if res.status_code == 200 and "candidates" in data:
                    reply = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"status": "ok", "reply": reply}
                else:
                    err_detail = data.get("error", {}).get("message", res.text)
                    last_error = f"[{model}] {err_detail}"
            except Exception as e:
                last_error = str(e)

        return {"status": "error", "message": f"Error de Google: {last_error}"}
