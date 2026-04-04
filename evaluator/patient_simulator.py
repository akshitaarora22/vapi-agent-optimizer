"""
evaluator/patient_simulator.py

Simulates realistic dental patient callers using Claude.
Each persona has different characteristics that stress-test the agent.
The simulator drives a back-and-forth conversation with the Vapi assistant
by injecting patient turns into a web call session.

NOTE: Since Vapi web calls are real-time audio, we simulate the conversation
by running it directly through the Anthropic API with the same system prompt,
then use Vapi's /call endpoint with a pre-scripted transcript injection
(using Vapi's `assistantOverrides` and a custom first message per persona).

For a fully automated loop without real audio, we drive the conversation
entirely via the Anthropic API and score the transcript ourselves.
This is the "shadow mode" approach — valid for optimization, clearly documented.
"""

import os
import anthropic
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

PATIENT_PERSONAS = [
    {
        "name": "DirectDave",
        "description": "Busy professional, knows exactly what he wants",
        "first_message": "Hi, I need to book a teeth cleaning for next Tuesday morning.",
        "behavior": (
            "You are Dave, a busy software engineer. You want to book a routine cleaning. "
            "You know your schedule: Tuesday or Wednesday morning works. "
            "Your name is Dave Chen, phone is 415-555-0192. "
            "Be brief and slightly impatient. Answer questions directly. "
            "If the agent is too slow or asks redundant questions, say 'can we speed this up?'"
        ),
    },
    {
        "name": "UncertainUma",
        "description": "First-time patient, unsure about what service she needs",
        "first_message": "Um, hi, I think I need to make an appointment? I'm not sure what kind though.",
        "behavior": (
            "You are Uma, calling a dental office for the first time. "
            "You have mild tooth pain but aren't sure if it's a cavity or just sensitivity. "
            "You're a bit nervous and hesitant. You'll answer questions but sometimes need clarification. "
            "Your name is Uma Patel. You're free most afternoons next week. "
            "Phone: 650-555-0847. You respond better when given options to choose from."
        ),
    },
    {
        "name": "ReschedulingRaj",
        "description": "Existing patient who needs to reschedule",
        "first_message": "Hey, I have an appointment next Monday but I need to change it.",
        "behavior": (
            "You are Raj, an existing patient. You had an appointment for Monday at 10am "
            "but a work conflict came up. You want to reschedule to the same week if possible, "
            "preferably Friday. Your name is Raj Sharma. You're a bit frustrated but polite. "
            "Phone: 408-555-0331. If Friday doesn't work, Thursday afternoon is fine."
        ),
    },
    {
        "name": "VagueVictor",
        "description": "Caller who gives minimal information and needs prompting",
        "first_message": "Yeah I want an appointment.",
        "behavior": (
            "You are Victor, a man of few words. You want a whitening appointment "
            "but you won't volunteer information unless asked directly. "
            "Answer each question with as few words as possible. "
            "Name: Victor Mills. Available: 'whenever'. Phone: 'sure' (then give 510-555-0274 if pushed). "
            "You're not rude, just very terse."
        ),
    },
    {
        "name": "EmergencyElena",
        "description": "Patient with urgent tooth pain needing same-day care",
        "first_message": "I'm in a lot of pain, I think I need emergency dental care today.",
        "behavior": (
            "You are Elena, experiencing acute tooth pain since this morning. "
            "You're distressed but coherent. You need same-day emergency care. "
            "Name: Elena Rodriguez. You can come in any time today. "
            "Phone: 415-555-0763. If they can't see you today, ask about the earliest possible slot."
        ),
    },
]


@dataclass
class ConversationResult:
    persona_name: str
    transcript: List[Dict[str, str]]  # [{"role": "assistant"|"user", "content": str}]
    appointment_booked: bool
    num_turns: int
    notes: str
    vapi_call_id: str = ""  # filled in by scorer after Vapi submission


def simulate_conversation(
    system_prompt: str,
    persona: Dict,
    max_turns: int = 12,
) -> ConversationResult:
    """
    Run a full simulated conversation between the dental agent (system_prompt)
    and a patient persona, both driven by Claude.
    Returns the full transcript and outcome.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Agent side: uses the optimized system prompt
    agent_messages = []

    # Patient side: uses the persona behavior prompt
    patient_system = (
        f"{persona['behavior']}\n\n"
        "You are on a phone call with a dental office AI receptionist. "
        "Respond naturally as this person would. "
        "Keep responses short (1-3 sentences). "
        "When the appointment is fully confirmed (agent says 'Your appointment has been scheduled'), "
        "say 'Great, thank you!' and end the conversation. "
        "If after 8 turns no progress is being made, politely end the call."
    )
    patient_messages = []

    transcript = []
    appointment_booked = False

    # Agent speaks first (its firstMessage)
    agent_first = "Hello, thank you for calling Bright Smile Dental. How can I help you today?"
    transcript.append({"role": "assistant", "content": agent_first})

    # Patient responds with their opening line
    current_patient_turn = persona["first_message"]
    transcript.append({"role": "user", "content": current_patient_turn})

    for turn in range(max_turns):
        # --- Agent turn ---
        agent_messages = _build_messages_from_transcript(transcript, perspective="agent")
        agent_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=system_prompt,
            messages=agent_messages,
        )
        agent_text = agent_resp.content[0].text.strip()
        transcript.append({"role": "assistant", "content": agent_text})

        # Check if agent confirmed booking
        if "appointment has been scheduled" in agent_text.lower():
            appointment_booked = True

        # --- Patient turn ---
        patient_messages = _build_messages_from_transcript(transcript, perspective="patient")
        patient_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            system=patient_system,
            messages=patient_messages,
        )
        patient_text = patient_resp.content[0].text.strip()
        transcript.append({"role": "user", "content": patient_text})

        # Check if patient ended call
        if any(phrase in patient_text.lower() for phrase in ["thank you!", "goodbye", "bye", "have a good"]):
            break

        if appointment_booked:
            break

    return ConversationResult(
        persona_name=persona["name"],
        transcript=transcript,
        appointment_booked=appointment_booked,
        num_turns=len([t for t in transcript if t["role"] == "assistant"]),
        notes="",
    )


def _build_messages_from_transcript(
    transcript: List[Dict[str, str]], perspective: str
) -> List[Dict[str, str]]:
    """
    Convert transcript to messages array from the given perspective.
    perspective="agent": assistant=agent, user=patient
    perspective="patient": assistant=patient, user=agent
    """
    messages = []
    for turn in transcript:
        if perspective == "agent":
            role = turn["role"]  # already "assistant" or "user"
        else:
            # Flip roles for the patient's perspective
            role = "user" if turn["role"] == "assistant" else "assistant"
        # Merge consecutive same-role messages
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += " " + turn["content"]
        else:
            messages.append({"role": role, "content": turn["content"]})
    return messages


def get_eval_personas(n: int = 5) -> List[Dict]:
    """Return n personas for evaluation. Always includes all 5 if n>=5."""
    if n >= len(PATIENT_PERSONAS):
        return PATIENT_PERSONAS
    return random.sample(PATIENT_PERSONAS, n)
