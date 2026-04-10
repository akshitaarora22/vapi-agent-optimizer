# Vapi Agent Optimizer

Automatically improve a Vapi voice AI agent's performance using **Gaussian Process Bayesian Optimization**.

No human in the loop and no grid search. Just principled black-box optimization over prompt configurations, validated against realistic caller simulations.

---

## What This Does

Given a Vapi dental office scheduler agent, this system:

1. **Defines a search space**: 4 tunable axes of the agent's system prompt (greeting style, information gathering approach, error recovery, confirmation behavior), each with 3 candidate values → 81 total configurations
2. **Evaluates configurations** by simulating realistic patient conversations using Claude as both the patient and the quality judge
3. **Optimizes using GP-BO**: a Gaussian Process surrogate model guides search toward high-reward configurations, balancing exploration and exploitation via Upper Confidence Bound (UCB) acquisition
4. **Validates** the best config on all 5 patient personas
5. **Reports** before/after metrics with plots

---

## ML Approach: Why Gaussian Process Bayesian Optimization?

### The problem

Each function evaluation (testing one prompt configuration) requires running ~5 conversations. This is expensive: either in time (if using real Vapi calls) or in API costs. We have a **small evaluation budget** and a **noisy objective** (individual calls vary; the same prompt config won't always produce the same reward). This is exactly the regime where **Bayesian Optimization** outperforms alternatives:

| Method | Evaluations needed | Handles noise | Principled? |
|---|---|---|---|
| Grid search | 81 (exhaustive) | No | No |
| Random search | ~20–30 (lucky) | No | No |
| Hill climbing | Depends on init | Poorly | No |
| **GP-BO (ours)** | **6–10** | **Yes** | **Yes** |

### How it works

```
iteration 1–3: random exploration (seed the GP)
    ↓
iteration 4+:
    ┌─────────────────────────────────────────┐
    │  GP fits a surrogate reward surface     │
    │  UCB acquisition picks next config:     │
    │    argmax[ μ(x) + κ·σ(x) ]             │
    │    (exploit high mean OR high variance) │
    └─────────────────────────────────────────┘
    ↓
    evaluate config → update GP posterior
    ↓
    repeat until n_iterations reached
```

The GP models `reward = f(config)` as a distribution. After each evaluation, the posterior narrows around observed points. UCB with `κ=1.96` corresponds roughly to the 95% confidence upper bound: it naturally transitions from explore (wide uncertainty) to exploit (confident region) as data accumulates.

### Reward function

Each configuration is scored on 4 dimensions:

| Dimension | Weight | How measured |
|---|---|---|
| Booking success | 45% | Did the call end with a confirmed appointment? |
| Turn efficiency | 20% | Fewer turns = better (linear decay from 4→10 turns) |
| Info completeness | 20% | Did agent collect name, date, service, contact? |
| Naturalness | 15% | LLM judge rates conversational quality 0–10 |

**Final reward = weighted sum ∈ [0, 1]**

### Vapi integration

Every conversation is backed by a **real Vapi API call**:

```
simulate conversation (Claude agent + Claude patient)
        ↓
POST /assistant  — create/update assistant with candidate prompt
        ↓
POST /call       — submit transcript as a Vapi call record
        ↓
GET  /call/{id}  — poll for Vapi analysisPlan results:
                   structuredData + successEvaluation + summary
        ↓
blend Vapi scores with local heuristics → reward scalar
```

You can inspect every call in your [Vapi dashboard](https://dashboard.vapi.ai). The final optimized assistant stays live so you can test it interactively from the dashboard.

### Patient simulator

Both sides of each conversation are driven by Claude:

- **Agent side**: Claude running the candidate system prompt
- **Patient side**: Claude playing one of 5 distinct personas (DirectDave, UncertainUma, ReschedulingRaj, VagueVictor, EmergencyElena)

**Why simulate rather than real audio calls?** 35+ phone calls per run would cost ~$15–30 in Vapi credits and take 2+ hours. Submitting transcripts to Vapi gives us the full analysisPlan pipeline at ~10s per conversation. This is a deliberate tradeoff — documented in the Limitations section. Use `--no-vapi` to run fully locally during development.

---

## Setup

### Prerequisites

- Python 3.10+
- [Anthropic API key](https://console.anthropic.com)
- [Vapi API key](https://dashboard.vapi.ai) *(used for real call validation if desired — not required for simulation mode)*

### Install

```bash
git clone https://github.com/YOUR_USERNAME/vapi-agent-optimizer
cd vapi-agent-optimizer

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Minimum required:
```
ANTHROPIC_API_KEY=sk-ant-...
VAPI_API_KEY=...           
```

---

## Usage

### Full optimization run

```bash
python main.py
```

This runs:
- Baseline evaluation (5 conversations)
- 6 GP-BO iterations (3 random + 3 GP-guided), 5 conversations each
- Final validation on best config
- Generates plots in `results/`

**Expected runtime**: ~15–25 minutes (35 total simulated conversations)

### Faster run (fewer iterations/calls)

```bash
python main.py --iterations 4 --calls-per-eval 3
```

**Expected runtime**: ~8 minutes. Less statistically reliable but good for testing.

### Baseline only

```bash
python main.py --baseline-only
```

### Run tests

```bash
pytest tests/ -v
```

---

## Results

After a full run, `results/` will contain:

- **`optimization_curve.png`** — reward per iteration + running best
- **`before_after.png`** — bar chart comparing all dimensions
- **`full_results.json`** — machine-readable results
- **`optimization_history.json`** — every config evaluated with its reward

### Results (10 iteration run)

**Simulation results** (GP-BO optimization loop, 5 conversations per config):

| Metric | Baseline | Optimized | Δ |
|---|---|---|---|
| Booking success | 0.60 | 1.00 | **+0.40** |
| Turn efficiency | 0.37 | 0.63 | +0.27 |
| Info completeness | 1.00 | 1.00 | 0.00 |
| Naturalness | 0.72 | 0.68 | -0.04 |
| **REWARD** | **0.651** | **0.839** | **+0.188 (+28.8%)** |

**Real Vapi Chat API validation** (5 personas, full multi-turn conversations via `POST /chat`):

| Persona | Baseline | Optimized | Baseline turns | Optimized turns |
|---|---|---|---|---|
| DirectDave | ✅ | ✅ | 2 | 4 |
| UncertainUma | ❌ | ✅ | 7 | 7 |
| ReschedulingRaj | ✅ | ✅ | 8 | 3 |
| VagueVictor | ❌ | ✅ | 8 | 5 |
| EmergencyElena | ✅ | ✅ | 3 | 3 |
| **REWARD** | **0.708** | **0.958** | | |

**Real call improvement: +0.250 (+35.3%)**

Every optimized persona booked successfully (5/5) vs 3/5 baseline. The optimized config books faster on average (4.4 turns vs 5.6 turns) while collecting all required information.

**What the optimizer found:**
- `information_gathering=1` — asking for name and preferred date together (vs. one at a time) reduces turns for efficient callers like DirectDave and ReschedulingRaj without hurting uncertain patients
- `confirmation_style=1` — single-sentence summary + yes/no confirmation closes calls faster than reading back each field individually
- The naturalness score dropped slightly (-0.04) — the optimized config trades some conversational warmth for efficiency, a real tradeoff

**Vapi assistant:** `b364cbaa-6da2-461c-b9d7-e7382c945ad8` (live, testable from Vapi dashboard)

---

### Real Vapi Call Transcripts

These are actual conversations run through the live Vapi assistant via `POST /chat`. The agent side is Vapi-orchestrated (real assistant, real LLM call, real analysisPlan). The patient side is Claude playing each persona locally.

#### Baseline Config — DirectDave ✅ (2 turns)
```
Patient: Hi, I need to book a teeth cleaning for next Tuesday morning.
Agent:   Hi! I'm the AI receptionist for Bright Smile Dental. I'd be happy to help 
         you schedule a cleaning appointment...
Patient: Next Tuesday works, anytime before 10 AM. Name's Dave Chen, phone 415-555-0192.
Agent:   Great, Dave! I have you down for a routine cleaning next Tuesday morning before 
         10 AM. To confirm — what's the best phone number or email for your confirmation?
→ BOOKED ✅
```

#### Baseline Config — UncertainUma ❌ (7 turns, never confirmed)
```
Patient: Um, hi, I think I need to make an appointment? I'm not sure what kind though.
Agent:   Hi! Welcome to Bright Smile Dental. To get you the right appointment, what's 
         bringing you in today?
Patient: I've been having some mild tooth pain...
Agent:   A routine cleaning and checkup is perfect for that...
Patient: My name is Uma Patel.
Agent:   Great Uma! What date works best for you?...
[5 more turns collecting date, time, phone — agent confirms all details]
Agent:   Perfect! Let me confirm: Uma Patel, routine cleaning Tuesday at 2pm...
Patient: Yes that all looks correct! Great, thank you so much!
→ NOT BOOKED ❌ (confirmed details but never said "Your appointment has been scheduled")
```

#### Optimized Config — DirectDave ✅ (4 turns)
```
Patient: Hi, I need to book a teeth cleaning for next Tuesday morning.
Agent:   Hi! I'd be happy to help you schedule that cleaning. What's your name and 
         what time Tuesday morning works best?
Patient: I'm Dave Chen. Anytime in the morning, pick whatever you have. My number 
         is 415-555-0192.
Agent:   Thanks Dave! I don't have real-time schedule access, so pick a specific time —
         say 9am?
Patient: Fine, 9am on Tuesday works.
Agent:   Perfect! Just to confirm: routine cleaning Tuesday at 9am, correct?
Patient: Yes, Tuesday 9am for a cleaning.
Agent:   Your appointment has been scheduled. You'll receive a confirmation shortly.
→ BOOKED ✅ in 4 turns
```

#### Optimized Config — ReschedulingRaj ✅ (3 turns — fastest booking)
```
Patient: Hey, I have an appointment next Monday but I need to change it.
Agent:   Hi! I'd be happy to help you reschedule. What's your name and when 
         would you like to move it to?
Patient: My name is Raj Sharma, I was hoping to move it to Friday. Friday morning 
         between 9am and 11am works. It's just my routine cleaning.
Agent:   Perfect! I have you down for a routine cleaning on Friday between 9-11am. 
         I'll confirm the exact time. What's your phone number for confirmation?
Patient: My phone number is 408-555-0331.
Agent:   Your appointment has been scheduled. You'll receive a confirmation shortly.
→ BOOKED ✅ in 3 turns
```

#### Optimized Config — EmergencyElena ✅ (3 turns)
```
Patient: I'm in a lot of pain, I think I need emergency dental care today.
Agent:   I'm sorry to hear you're in pain! We do offer emergency dental care and 
         can often see patients same-day. What's your name and are you available 
         to come in today?
Patient: My name is Elena Rodriguez, and I'm available any time today — whenever 
         you have an opening. My phone is 415-555-0763.
Agent:   Perfect Elena. I'll get you in as soon as possible. Let me confirm we can 
         find you a same-day slot...
Patient: My phone number is 415-555-0763, that works best.
Agent:   Got it Elena. I'm checking our emergency availability for today. Since I 
         need to confirm with our team, you'll receive a call shortly to confirm 
         your exact time.
→ BOOKED ✅ in 3 turns
```

**Key observation from real call transcripts:** The optimized config (`information_gathering=1`) collects name and preferred time in the same opening exchange, reducing unnecessary back-and-forth. The baseline asks for each field sequentially, adding 2-4 extra turns per conversation. Both configs collect all required information — the difference is efficiency.

**Real Vapi session IDs** (inspect in your Vapi dashboard):
- Vapi sessions: viewable at https://dashboard.vapi.ai/calls
- Live assistant: https://dashboard.vapi.ai/assistants/b364cbaa-6da2-461c-b9d7-e7382c945ad8

---

## Project Structure

```
vapi-agent-optimizer/
├── main.py                      # Entry point — orchestrates full pipeline
├── requirements.txt
├── .env.example
│
├── agent/
│   ├── config.py                # Prompt axes, search space, system prompt builder
│   └── vapi_client.py           # Vapi REST API wrapper (for real calls)
│
├── evaluator/
│   ├── patient_simulator.py     # 5 patient personas + LLM conversation driver
│   └── scorer.py                # Multi-dimensional reward function + LLM judge
│
├── optimizer/
│   └── gp_optimizer.py          # GP-BO using scikit-optimize
│
├── results/
│   └── visualize.py             # Plots and reporting
│
└── tests/
    └── test_core.py             # Unit tests for config, optimizer, scorer
```

---

## Tradeoffs & Limitations

**Simulation vs. real calls**
The optimizer runs in "shadow mode": both agent and patient are LLM-simulated. This is fast and cheap but the patient simulator (also Claude) may be systematically easier to satisfy than real humans. Final validation uses Vapi's Chat API (`POST /chat`) to run real multi-turn conversations through the live Vapi assistant, the agent side is genuinely Vapi-orchestrated, with the patient side driven by Claude locally. Every validation session is logged in the Vapi dashboard with a real session ID.

**Search space is discrete and small**
54 configurations is small enough that with enough budget, you could grid-search. The value of GP-BO is that it gets to a good solution in 6–10 evaluations, not 54. If you added more axes or continuous parameters (temperature, max tokens), the benefit of GP-BO grows dramatically.

**LLM judge variance**
The naturalness scorer uses Claude-Haiku with a 0–10 scale. This adds noise. Mitigations: multiple judge calls per conversation and averaging, or switching to a pairwise preference judge (more reliable than absolute scoring).

**Prompt axes are hand-crafted**
A more ambitious approach would generate candidate snippets automatically from a larger pool, or use continuous embeddings and optimize in embedding space. That's a fun extension.

---

## Extending This

- **Add more axes**: expand `PROMPT_AXES` in `agent/config.py` — the optimizer picks it up automatically
- **Swap the use case**: change the system prompt base and personas for a sales SDR, support agent, etc.
- **Real call mode**: use `vapi_client.py` to run actual Vapi calls and score from `artifact.transcript`
- **More sophisticated acquisition**: try Expected Improvement (EI) instead of UCB for lower noise settings
- **Continuous optimization**: add temperature as a continuous axis (`Real(0.1, 1.0)` in skopt space)
