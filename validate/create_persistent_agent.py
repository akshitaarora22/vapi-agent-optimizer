import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
from agent.config import BASELINE_CONFIG, build_system_prompt
from agent.vapi_client import VapiClient

vapi = VapiClient()
aid = vapi.create_assistant(build_system_prompt(BASELINE_CONFIG), "DentalAgent-Persistent")
print("Your persistent agent ID:", aid)
print()
print("Next steps:")
print(f"  1. Add to .env:  PERSISTENT_AGENT_ID={aid}")
print(f"  2. Add to .env:  VAPI_AGENT_PHONE_NUMBER=+17348080512")
print(f"  3. Vapi dashboard -> Phone Numbers -> +17348080512 -> Inbound Assistant -> {aid}")
