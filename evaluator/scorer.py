"""
evaluator/scorer.py

Scores a conversation transcript on multiple dimensions,
then combines them into a single scalar reward for the optimizer.

Scoring pipeline
----------------
1. Run simulated conversation (Claude agent + Claude patient)
2. Submit transcript to Vapi via POST /call  → Vapi runs its analysisPlan
3. Poll GET /call/{id} for Vapi's structuredData + successEvaluation
4. Blend Vapi's scores with local heuristic scores into a final reward

This means every evaluation is backed by a real Vapi call object you can
inspect in the dashboard, and Vapi's own analysis contributes to the reward.

Scoring dimensions:
  1. booking_success   (0 or 1)   — Vapi successEvaluation (primary) OR local heuristic
  2. turn_efficiency   (0–1)      — fewer turns = better
  3. info_completeness (0–1)      — Vapi structuredData fields + local heuristic
  4. naturalness       (0–1)      — Claude LLM judge

Final reward = weighted sum, in [0, 1].
"""

import os
import json
import anthropic
from typing import List, Dict, Optional, TYPE_CHECKING
from evaluator.patient_simulator import ConversationResult

if TYPE_CHECKING:
    from agent.vapi_client import VapiClient

# Weights must sum to 1.0
SCORE_WEIGHTS = {
    "booking_success": 0.45,
    "turn_efficiency": 0.20,
    "info_completeness": 0.20,
    "naturalness": 0.15,
}

MAX_ACCEPTABLE_TURNS = 10


def score_conversation(
    result: ConversationResult,
    vapi_client: Optional["VapiClient"] = None,
    assistant_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Score a single conversation result.

    If vapi_client and assistant_id are provided:
      - Submits transcript to Vapi (creates a real call record)
      - Waits for Vapi's analysisPlan to run
      - Uses Vapi's successEvaluation as the primary booking signal
      - Blends Vapi's structuredData into info_completeness

    Falls back gracefully to local heuristics if Vapi is unavailable.
    """
    scores = {}
    vapi_scores = {}
    call_id = None

    # --- Submit to Vapi and get analysis ---
    if vapi_client and assistant_id:
        try:
            call_id = vapi_client.create_call_with_transcript(
                assistant_id=assistant_id,
                transcript_turns=result.transcript,
                persona_name=result.persona_name,
            )
            result.vapi_call_id = call_id

            # Poll for Vapi analysis (up to 45s)
            call_data = vapi_client.wait_for_analysis(call_id, timeout=45)
            vapi_scores = vapi_client.extract_vapi_scores(call_data)
        except Exception:
            vapi_scores = {}  # Vapi logging failed — continue with local scoring

    # 1. Booking success — prefer Vapi's evaluation, fall back to local
    if vapi_scores.get("vapi_success") is not None:
        scores["booking_success"] = 1.0 if vapi_scores["vapi_success"] else 0.0
    else:
        scores["booking_success"] = 1.0 if result.appointment_booked else 0.0

    # 2. Turn efficiency
    turns = result.num_turns
    scores["turn_efficiency"] = max(0.0, 1.0 - (turns - 4) / (MAX_ACCEPTABLE_TURNS - 4))

    # 3. Info completeness — blend Vapi structured data with local heuristic
    local_completeness = _score_info_completeness(result.transcript)
    if vapi_scores.get("vapi_structured"):
        vapi_completeness = _score_vapi_structured(vapi_scores["vapi_structured"])
        # Weighted blend: 60% Vapi, 40% local heuristic
        scores["info_completeness"] = 0.6 * vapi_completeness + 0.4 * local_completeness
    else:
        scores["info_completeness"] = local_completeness

    # 4. Naturalness — Claude LLM judge (independent of Vapi)
    scores["naturalness"] = _score_naturalness(result.transcript, result.persona_name)

    # Final weighted reward
    scores["reward"] = sum(SCORE_WEIGHTS[dim] * scores[dim] for dim in SCORE_WEIGHTS)

    # Attach Vapi metadata for reporting
    scores["vapi_call_id"] = call_id or ""
    scores["vapi_success_raw"] = vapi_scores.get("vapi_success", None)

    return scores


def _score_vapi_structured(structured: Dict) -> float:
    """
    Score info completeness using Vapi's structuredData extraction.
    Returns fraction of required fields that were successfully extracted.
    """
    required_fields = {
        "appointment_booked": lambda v: v is not None,
        "patient_name": lambda v: bool(v),
        "preferred_date": lambda v: bool(v),
        "contact_info_collected": lambda v: v is True,
    }
    filled = sum(
        1 for field, check in required_fields.items()
        if check(structured.get(field))
    )
    return filled / len(required_fields)


def _score_info_completeness(transcript: List[Dict[str, str]]) -> float:
    """Local heuristic: did the agent ask for all required fields?"""
    agent_text = " ".join(
        t["content"].lower() for t in transcript if t["role"] == "assistant"
    )
    signals = {
        "name": any(w in agent_text for w in ["name", "who", "patient"]),
        "date": any(w in agent_text for w in ["date", "when", "day", "time", "schedule"]),
        "service": any(w in agent_text for w in ["service", "cleaning", "whitening", "filling", "type of", "what brings"]),
        "contact": any(w in agent_text for w in ["phone", "email", "number", "contact", "reach"]),
    }
    return sum(signals.values()) / len(signals)


def _score_naturalness(transcript: List[Dict[str, str]], persona_name: str) -> float:
    """Claude LLM judge for conversational naturalness, 0–1."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    formatted = "\n".join(
        f"{'AGENT' if t['role'] == 'assistant' else 'PATIENT'}: {t['content']}"
        for t in transcript
    )
    judge_prompt = (
        "You are evaluating a dental office AI receptionist. "
        "Read this call transcript and rate the AGENT's naturalness on a scale of 0-10.\n\n"
        f"Transcript:\n{formatted}\n\n"
        "Scoring guide: 9-10=perfectly natural human-like, 7-8=mostly natural minor issues, "
        "5-6=functional but robotic, 3-4=repetitive or awkward, 0-2=broken.\n\n"
        "Reply with ONLY valid JSON, no extra text: {\"score\": 7, \"reason\": \"one sentence\"}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": judge_prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extract JSON object if there's surrounding text
        import re
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            raw = match.group(0)
        data = json.loads(raw)
        score = float(data["score"])
        return min(1.0, max(0.0, score / 10.0))
    except Exception as e:
        # Try a simpler fallback prompt
        try:
            resp2 = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": f"Rate this dental receptionist AI from 0-10 for naturalness. Reply with just the number.\n\n{formatted[:500]}"
                }],
            )
            score = float(resp2.content[0].text.strip().split()[0])
            return min(1.0, max(0.0, score / 10.0))
        except Exception:
            return 0.5


def score_multiple_conversations(
    results: List[ConversationResult],
    vapi_client: Optional["VapiClient"] = None,
    assistant_id: Optional[str] = None,
) -> Dict[str, float]:
    """
    Score a batch of conversations and return averaged scores.
    Passes vapi_client through so each conversation gets a real Vapi call record.
    """
    all_scores = [
        score_conversation(r, vapi_client=vapi_client, assistant_id=assistant_id)
        for r in results
    ]

    # Average numeric fields, skip string metadata
    numeric_keys = [k for k in all_scores[0] if isinstance(all_scores[0][k], (int, float))]
    averaged = {k: sum(s[k] for s in all_scores) / len(all_scores) for k in numeric_keys}

    # Collect call IDs for reporting
    averaged["vapi_call_ids"] = [s.get("vapi_call_id", "") for s in all_scores]

    return averaged


def format_scores(scores: Dict[str, float], config_axes: Dict[str, int]) -> str:
    """Pretty-print scores for console output."""
    lines = [
        f"  booking_success:   {scores.get('booking_success', 0):.2f}",
        f"  turn_efficiency:   {scores.get('turn_efficiency', 0):.2f}",
        f"  info_completeness: {scores.get('info_completeness', 0):.2f}",
        f"  naturalness:       {scores.get('naturalness', 0):.2f}",
        f"  ── REWARD:         {scores.get('reward', 0):.4f}",
    ]
    call_ids = scores.get("vapi_call_ids", [])
    if call_ids and any(call_ids):
        lines.append(f"  vapi_calls:        {len([c for c in call_ids if c])} logged to Vapi")
    return "\n".join(lines)
