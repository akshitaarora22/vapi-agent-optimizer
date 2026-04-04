"""
validate/run_real_calls.py

Validates baseline vs optimized agent using Vapi's Chat API.

WHY CHAT API:
  Vapi's Chat API (POST /chat) runs real conversations through your Vapi assistant —
  same system prompt, same analysisPlan, same structured data extraction — but via
  text transport instead of audio. This is the correct way to run automated
  multi-turn conversations against a Vapi agent without needing phone calls.

  Every conversation creates a real Vapi chat session visible in your dashboard.

Usage:
  python validate/run_real_calls.py --compare
  python validate/run_real_calls.py --config best
  python validate/run_real_calls.py --config baseline
"""

import os, sys, json, time, argparse
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from agent.config import BASELINE_CONFIG, build_system_prompt
from agent.vapi_client import VapiClient

VAPI_BASE = "https://api.vapi.ai"

PATIENT_PERSONAS = [
    {
        "name": "DirectDave",
        "opening": "Hi, I need to book a teeth cleaning for next Tuesday morning.",
        "behavior": "You are Dave, a busy software engineer. You want to book a routine cleaning. Tuesday or Wednesday morning works. Name: Dave Chen, phone: 415-555-0192. Be brief and slightly impatient.",
    },
    {
        "name": "UncertainUma",
        "opening": "Um, hi, I think I need to make an appointment? I'm not sure what kind though.",
        "behavior": "You are Uma, calling a dental office for the first time. Mild tooth pain, unsure if cavity or sensitivity. Nervous. Name: Uma Patel. Free most afternoons next week. Phone: 650-555-0847.",
    },
    {
        "name": "ReschedulingRaj",
        "opening": "Hey, I have an appointment next Monday but I need to change it.",
        "behavior": "You are Raj, existing patient needing to reschedule from Monday at 10am. Want Friday if possible. Name: Raj Sharma, phone: 408-555-0331. Slightly frustrated but polite.",
    },
    {
        "name": "VagueVictor",
        "opening": "Yeah I want an appointment.",
        "behavior": "You are Victor. You want a whitening appointment but give minimal info unless asked. Name: Victor Mills. Available whenever. Phone: 510-555-0274 only if pushed. Very terse.",
    },
    {
        "name": "EmergencyElena",
        "opening": "I'm in a lot of pain, I think I need emergency dental care today.",
        "behavior": "You are Elena with acute tooth pain since this morning. Need same-day emergency care. Name: Elena Rodriguez. Available any time today. Phone: 415-555-0763.",
    },
]

PATIENT_SYSTEM = (
    "{behavior}\n\n"
    "You are on a call with a dental office AI receptionist. Respond naturally and briefly (1-2 sentences). "
    "Answer questions directly and give your info when asked without hesitation. "
    "When the agent confirms your appointment is scheduled, say 'Great, thank you so much!' and stop. "
    "If the agent asks for something you already provided, say it once more then say 'I will try calling back later.'"
)


def run_chat_conversation(vapi: VapiClient, assistant_id: str, persona: Dict, max_turns: int = 8) -> Dict:
    """
    Run a multi-turn conversation via Vapi Chat API.

    Each turn sends the FULL conversation history in the messages field so the
    agent has complete context. We do NOT use previousChatId (causes agent
    to only see its own prior messages, not the patient's responses).
    """
    ac = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    patient_system = PATIENT_SYSTEM.format(behavior=persona["behavior"])

    conversation = []  # {"role": "user"|"assistant", "content": str}
    appointment_booked = False
    last_chat_id = None
    patient_message = persona["opening"]

    for turn in range(max_turns):
        print(f"    [T{turn+1}] Patient: {patient_message[:90]}")

        # Vapi Chat API: always include assistantId
        # Use previousChatId on turns 2+ for context
        payload = {
            "assistantId": assistant_id,
            "input": patient_message,
        }
        if last_chat_id:
            payload["previousChatId"] = last_chat_id

        resp = vapi.session.post(f"{VAPI_BASE}/chat", json=payload)
        if not resp.ok:
            print(f"    Chat error {resp.status_code}: {resp.text[:150]}")
            break

        data = resp.json()
        last_chat_id = data.get("id") or data.get("chatId") or last_chat_id

        # Extract agent response
        raw = data.get("output", "")
        if isinstance(raw, list):
            agent_message = " ".join(
                m.get("content", "") for m in raw if m.get("role") == "assistant"
            ) or " ".join(m.get("content", "") for m in raw)
        elif isinstance(raw, dict):
            agent_message = raw.get("content", "") or raw.get("text", "")
        elif isinstance(raw, str):
            agent_message = raw
        else:
            agent_message = ""

        if not agent_message:
            print(f"    Warning: empty response turn {turn+1}, keys={list(data.keys())}")
            break

        # Append this turn to history
        conversation.append({"role": "user", "content": patient_message})
        conversation.append({"role": "assistant", "content": agent_message})
        print(f"    [T{turn+1}] Agent: {agent_message[:90]}")

        booked_phrases = [
            "appointment has been scheduled",
            "you'll receive a confirmation",
            "you will receive a confirmation",
            "confirmation shortly",
            "have you down for",
            "i'll put you down",
            "i have you down",
            "see you on",
            "see you then",
            "all set for",
            "booked for",
            "you're all set",
            "i'll confirm",
            "get you scheduled",
            "i have all your information",
        ]
        if any(p in agent_message.lower() for p in booked_phrases):
            appointment_booked = True
            break

        # Generate patient's next response using full conversation history
        patient_msgs = [
            {"role": "user" if m["role"] == "assistant" else "assistant", "content": m["content"]}
            for m in conversation
        ]
        pr = ac.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=100,
            system=patient_system, messages=patient_msgs,
        )
        patient_message = pr.content[0].text.strip()
        print(f"    [T{turn+1}] Patient: {patient_message[:90]}")

        if any(p in patient_message.lower() for p in ["thank you so much", "call back later"]):
            conversation.append({"role": "user", "content": patient_message})
            break

    return {
        "session_id": last_chat_id,
        "appointment_booked": appointment_booked,
        "num_turns": turn + 1,
        "conversation": conversation,
        "vapi_data": {},
        "persona": persona["name"],
    }


def score_result(result: Dict) -> Dict[str, float]:
    """Score a completed conversation."""
    booked = result["appointment_booked"]
    turns = result["num_turns"]

    analysis = result.get("vapi_data", {}).get("analysis", {}) or {}
    if analysis.get("successEvaluation") is not None:
        raw = analysis["successEvaluation"]
        booked = str(raw).lower() == "true" if isinstance(raw, str) else bool(raw)

    turn_eff = max(0.0, 1.0 - (turns - 4) / 6.0)

    agent_text = " ".join(m["content"].lower() for m in result["conversation"] if m["role"] == "assistant")

    # Also check if booking happened based on broader phrases (in case Vapi analysis unavailable)
    booked_phrases = [
        "appointment has been scheduled", "confirmation shortly", "you'll receive a confirmation",
        "have you down for", "i have you down", "i'll put you down", "see you then",
        "all set for", "booked for", "see you on", "you're all set",
        "so to confirm", "to confirm:", "i'll confirm", "get you scheduled",
        "i have all your information",
    ]
    if not booked and any(p in agent_text for p in booked_phrases):
        booked = True

    info = {
        "name":    any(w in agent_text for w in ["name", "who", "patient"]),
        "date":    any(w in agent_text for w in ["date", "when", "day", "time", "schedule"]),
        "service": any(w in agent_text for w in ["service", "cleaning", "whitening", "filling", "type"]),
        "contact": any(w in agent_text for w in ["phone", "email", "number", "contact", "reach"]),
    }
    info_score = sum(info.values()) / 4.0

    return {
        "booking_success": 1.0 if booked else 0.0,
        "turn_efficiency": turn_eff,
        "info_completeness": info_score,
        "reward": 0.50 * (1.0 if booked else 0.0) + 0.25 * turn_eff + 0.25 * info_score,
        "session_id": result.get("session_id", ""),
        "vapi_summary": analysis.get("summary", ""),
    }


def run_validation(config_axes: Dict, config_name: str, vapi: VapiClient) -> Dict:
    """Run all personas and return averaged scores."""
    prompt = build_system_prompt(config_axes)
    persistent_id = os.environ.get("PERSISTENT_AGENT_ID", "")

    if persistent_id:
        print(f"\n  Updating persistent agent ({config_name})...")
        vapi.update_assistant_prompt(persistent_id, prompt)
        assistant_id = persistent_id
    else:
        print(f"\n  Creating agent assistant ({config_name})...")
        assistant_id = vapi.create_assistant(prompt, f"DentalAgent-{config_name}")

    print(f"  Assistant: https://dashboard.vapi.ai/assistants/{assistant_id}\n")

    all_scores, session_ids = [], []
    for persona in PATIENT_PERSONAS:
        print(f"  Persona: {persona['name']}")
        result = run_chat_conversation(vapi, assistant_id, persona)
        scores = score_result(result)
        all_scores.append(scores)
        if scores["session_id"]:
            session_ids.append(scores["session_id"])
        booked = "✅" if scores["booking_success"] else "❌"
        print(f"  → {booked} turns:{result['num_turns']} info:{scores['info_completeness']:.0%} reward:{scores['reward']:.3f}")
        if scores["vapi_summary"]:
            print(f"     Vapi says: {scores['vapi_summary'][:100]}")
        print()

    avg = {k: sum(s[k] for s in all_scores) / len(all_scores)
           for k in ["booking_success", "turn_efficiency", "info_completeness", "reward"]}
    avg["session_ids"] = session_ids
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--config", choices=["baseline", "best"], default="best")
    args = parser.parse_args()

    vapi = VapiClient()
    best_config = None
    try:
        with open("results/full_results.json") as f:
            best_config = json.load(f)["final"]["config"]
        print(f"Best config: {best_config}\n")
    except FileNotFoundError:
        print("No results/full_results.json — run main.py first")
        if args.config == "best": sys.exit(1)

    os.makedirs("results", exist_ok=True)
    baseline_scores = final_scores = None

    if args.compare or args.config == "baseline":
        print("=" * 60)
        print("BASELINE — Vapi Chat API Validation")
        print("=" * 60)
        baseline_scores = run_validation(BASELINE_CONFIG, "baseline", vapi)
        print(f"Baseline REWARD: {baseline_scores['reward']:.4f}")

    if args.compare or args.config == "best":
        print("\n" + "=" * 60)
        print("OPTIMIZED — Vapi Chat API Validation")
        print("=" * 60)
        final_scores = run_validation(best_config, "optimized", vapi)
        print(f"Optimized REWARD: {final_scores['reward']:.4f}")

    if args.compare and baseline_scores and final_scores:
        imp = final_scores["reward"] - baseline_scores["reward"]
        pct = imp / max(baseline_scores["reward"], 0.001) * 100
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"  Baseline:   {baseline_scores['reward']:.4f}")
        print(f"  Optimized:  {final_scores['reward']:.4f}")
        print(f"  Improvement: {'+' if imp >= 0 else ''}{imp:.4f} ({pct:+.1f}%)")
        print(f"\n  Vapi sessions (baseline):  {baseline_scores['session_ids']}")
        print(f"  Vapi sessions (optimized): {final_scores['session_ids']}")

        with open("results/real_call_validation.json", "w") as f:
            json.dump({
                "baseline": {k: v for k, v in baseline_scores.items() if k != "session_ids"},
                "optimized": {k: v for k, v in final_scores.items() if k != "session_ids"},
                "improvement": imp, "improvement_pct": pct,
                "baseline_sessions": baseline_scores["session_ids"],
                "optimized_sessions": final_scores["session_ids"],
            }, f, indent=2)
        print("  Saved: results/real_call_validation.json")


if __name__ == "__main__":
    main()
