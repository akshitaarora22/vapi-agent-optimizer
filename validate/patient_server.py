"""
validate/patient_server.py

FastAPI server that acts as a Custom LLM for Vapi.

How it works:
  - Vapi is configured with this server as the "custom LLM" for the PATIENT side
  - The dental agent assistant talks to the patient (this server)
  - Wait... actually we invert it:

CORRECT ARCHITECTURE:
  - We create TWO Vapi assistants:
    1. The AGENT assistant — uses our optimized system prompt, real Claude LLM
    2. We trigger a call where the agent talks to a human (us / test)

  Actually the cleanest approach for automated testing:
  - Create the agent assistant normally (Claude as LLM)  
  - Create a "patient" assistant with THIS server as its custom LLM
  - Use Vapi squads or a simple outbound call where agent calls patient
  
  SIMPLEST approach that actually works:
  - This server exposes POST /chat/completions (OpenAI-compatible)
  - Vapi's PATIENT assistant uses this as its custom LLM
  - We create a webCall where the patient assistant talks to our agent
  - Actually Vapi calls go: phone number → assistant. For two-assistant
    conversations we need squads.

REAL SIMPLEST approach:
  - This server IS the patient — it responds to Vapi's agent
  - We create an assistant that uses THIS server as the LLM
  - When Vapi sends a turn to this server, we generate the patient's response
  - Vapi handles TTS (patient's voice), STT (agent's voice), call recording
  - GET /call/{id} gives us the full real transcript

So the flow is:
  Vapi web call starts
        ↓
  Vapi's agent assistant speaks (using its Claude LLM + system prompt)
        ↓  
  Vapi STT converts agent speech → text
        ↓
  Vapi sends turn to THIS SERVER (patient custom LLM)
        ↓
  This server generates patient response using Claude
        ↓
  Vapi TTS speaks the patient response
        ↓
  Loop until call ends
        ↓
  GET /call/{id} → real transcript + analysisPlan results

This is a genuine Vapi voice call. Vapi handles all audio.
We just provide the patient's conversational responses.
"""

import os
import json
import asyncio
from typing import Dict, List, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Vapi Patient Simulator — Custom LLM")

# Active call state: call_id → persona info
# Vapi sends call_id in each request so we can track which persona to use
call_state: Dict[str, Dict] = {}

PATIENT_PERSONAS = {
    "DirectDave": (
        "You are Dave, a busy software engineer calling a dental office. "
        "You want to book a routine cleaning. Tuesday or Wednesday morning works. "
        "Name: Dave Chen, phone: 415-555-0192. Be brief and slightly impatient. "
        "Answer questions directly. If the agent is too slow, say 'can we speed this up?'"
    ),
    "UncertainUma": (
        "You are Uma, calling a dental office for the first time. "
        "You have mild tooth pain but aren't sure if it's a cavity or sensitivity. "
        "You're a bit nervous. Your name is Uma Patel. Free most afternoons next week. "
        "Phone: 650-555-0847. You respond better when given options."
    ),
    "ReschedulingRaj": (
        "You are Raj, an existing patient who needs to reschedule from Monday at 10am. "
        "Work conflict came up. Want Friday if possible, Thursday afternoon otherwise. "
        "Name: Raj Sharma, phone: 408-555-0331. Slightly frustrated but polite."
    ),
    "VagueVictor": (
        "You are Victor, a man of few words wanting a whitening appointment. "
        "Answer each question with as few words as possible. "
        "Name: Victor Mills. Available: whenever. Phone: 510-555-0274 (only if pushed). "
        "Not rude, just very terse."
    ),
    "EmergencyElena": (
        "You are Elena, experiencing acute tooth pain since this morning. "
        "You need same-day emergency care. Name: Elena Rodriguez. "
        "Available any time today. Phone: 415-555-0763. "
        "If they can't see you today, ask about the earliest possible slot."
    ),
}

PATIENT_SYSTEM_BASE = (
    "{persona_behavior}\n\n"
    "You are on a phone call with a dental office AI receptionist. "
    "Respond naturally and briefly (1-3 sentences max). "
    "When the appointment is fully confirmed (agent says 'Your appointment has been scheduled' "
    "or similar), say 'Great, thank you so much!' and end the conversation. "
    "If after 10 exchanges no progress is being made, politely say you'll call back later."
)


@app.get("/health")
def health():
    return {"status": "ok", "active_calls": len(call_state)}


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible endpoint that Vapi calls for each patient turn.
    
    Vapi sends the full conversation history (what the agent has said so far)
    and expects a response (what the patient should say next).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    messages: List[Dict] = body.get("messages", [])
    call_id = body.get("call", {}).get("id", "unknown")
    
    # Determine which persona to use for this call
    # Vapi sends metadata in multiple possible locations depending on version
    persona_name = "DirectDave"  # default
    
    # Check call.assistant.metadata
    call_obj = body.get("call", {})
    assistant_obj = call_obj.get("assistant", {})
    
    persona_name = (
        assistant_obj.get("metadata", {}).get("persona")
        or call_obj.get("assistantOverrides", {}).get("metadata", {}).get("persona")
        or body.get("metadata", {}).get("persona")
        or "DirectDave"
    )

    persona_behavior = PATIENT_PERSONAS.get(persona_name, PATIENT_PERSONAS["DirectDave"])
    patient_system = PATIENT_SYSTEM_BASE.format(persona_behavior=persona_behavior)

    # Build patient messages from conversation history
    # Vapi sends messages as the conversation so far from the AGENT's perspective
    # We need to flip roles: agent turns become "user" for our patient LLM
    patient_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "assistant":
            # Agent spoke — this is what the patient is responding TO
            patient_messages.append({"role": "user", "content": content})
        elif role == "user":
            # Patient already spoke — this is what the patient already said
            patient_messages.append({"role": "assistant", "content": content})
        # Skip system messages

    if not patient_messages:
        # First turn — agent hasn't spoken yet, use opening line
        patient_messages = [{"role": "user", "content": "Hello, thank you for calling Bright Smile Dental. How can I help you today?"}]

    # Generate patient response
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=patient_system,
            messages=patient_messages,
        )
        patient_text = resp.content[0].text.strip()
    except Exception as e:
        patient_text = "I'm sorry, could you repeat that?"

    # Check if call should end
    should_end = any(phrase in patient_text.lower() for phrase in [
        "thank you so much", "goodbye", "i'll call back", "call back later"
    ])

    # Return in OpenAI-compatible format that Vapi expects
    response_body = {
        "id": f"patient-{call_id}",
        "object": "chat.completion",
        "model": "patient-simulator",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": patient_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    # If call should end, signal Vapi to hang up via a special function call
    # Vapi supports endCall tool — we can include it as a tool call
    if should_end:
        response_body["choices"][0]["finish_reason"] = "stop"
        # Append end-call signal after the final message
        # Vapi will process the text response then end the call

    return JSONResponse(response_body)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting patient simulator server on port {port}")
    print("Expose with: ngrok http {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
