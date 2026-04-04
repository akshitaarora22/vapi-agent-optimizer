"""
agent/config.py

Defines the prompt search space as discrete axes.
Each axis represents one tunable dimension of the agent's behavior.
The Bayesian optimizer will search over combinations of these axes.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Prompt axes — the "coordinates" of our search space
# Each axis has a name and a list of candidate snippets.
# The optimizer picks one value per axis each iteration.
# ---------------------------------------------------------------------------

PROMPT_AXES: Dict[str, List[str]] = {
    # How the agent opens the conversation
    "greeting_style": [
        "Brief and direct: introduce yourself and immediately ask how you can help.",
        "Warm and welcoming: introduce yourself, welcome the caller, and ask how you can assist them today.",
        "Professional and structured: introduce yourself, state the office name, and guide them to choose from booking, rescheduling, or questions.",
    ],

    # How verbose the agent is when asking for information
    "information_gathering": [
        "Ask for one piece of information at a time. Wait for the answer before proceeding.",
        "Ask for name and preferred date together to be efficient, then collect remaining details one by one.",
        "Collect all required information (name, date, service, contact) in a structured sequence, confirming each before moving on.",
    ],

    # How the agent handles unclear or ambiguous input
    "error_recovery": [
        "If you don't understand, ask the caller to repeat themselves once. If still unclear, offer to transfer to a human.",
        "If you don't understand, rephrase your question more specifically and offer a few example options to choose from.",
        "If you don't understand, apologize briefly, then break the question into smaller parts and try again.",
    ],

    # Confirmation behavior before finalizing a booking
    "confirmation_style": [
        "Before confirming the appointment, read back all details and ask the caller to confirm each one.",
        "Summarize the appointment details in one sentence and ask for a simple yes/no confirmation.",
        "Confirm the appointment immediately after collecting all details, then offer to make changes if needed.",
    ],
}

# Number of values per axis — used to define the search space bounds
AXIS_SIZES = {k: len(v) for k, v in PROMPT_AXES.items()}

# The fixed parts of the system prompt that never change
SYSTEM_PROMPT_BASE = """You are a friendly and professional AI receptionist for Bright Smile Dental, a dental office in San Francisco.

Your primary goal is to help patients schedule, reschedule, or cancel appointments efficiently and accurately.

Services offered:
- Routine cleaning and checkup (45 min)
- Teeth whitening (60 min)
- Cavity filling (30-60 min)
- Emergency dental care (immediate, by availability)
- Orthodontic consultation (30 min)

Available hours: Monday–Friday 8am–5pm, Saturday 9am–2pm. Closed Sunday.

When booking an appointment, you must collect:
1. Patient's full name
2. Preferred date and time
3. Type of service needed
4. Phone number or email for confirmation

{greeting_style_snippet}

{information_gathering_snippet}

{error_recovery_snippet}

{confirmation_style_snippet}

Always be concise. Keep responses under 3 sentences unless explaining something complex.
Never make up availability — tell the caller you will confirm their slot and they'll receive a follow-up.
End every successful booking by saying: "Your appointment has been scheduled. You'll receive a confirmation shortly."
"""


def build_system_prompt(config: Dict[str, int]) -> str:
    """
    Build a full system prompt from a configuration dict.
    config maps axis_name -> axis_index (int).
    """
    return SYSTEM_PROMPT_BASE.format(
        greeting_style_snippet=PROMPT_AXES["greeting_style"][config["greeting_style"]],
        information_gathering_snippet=PROMPT_AXES["information_gathering"][config["information_gathering"]],
        error_recovery_snippet=PROMPT_AXES["error_recovery"][config["error_recovery"]],
        confirmation_style_snippet=PROMPT_AXES["confirmation_style"][config["confirmation_style"]],
    )


# Default baseline config — index 0 for all axes (simplest options)
BASELINE_CONFIG: Dict[str, int] = {axis: 0 for axis in PROMPT_AXES}


@dataclass
class AgentConfig:
    """Represents a fully-specified agent configuration."""
    axes: Dict[str, int]
    system_prompt: str
    assistant_id: str = ""  # filled in after creating on Vapi

    @classmethod
    def from_axes(cls, axes: Dict[str, int]) -> "AgentConfig":
        return cls(axes=axes, system_prompt=build_system_prompt(axes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axes": self.axes,
            "assistant_id": self.assistant_id,
        }
