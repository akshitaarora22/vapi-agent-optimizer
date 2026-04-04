"""
main.py

Entry point for the Vapi Agent Optimizer.

Full pipeline:
  1. Create a real Vapi assistant via POST /assistant
  2. Evaluate baseline: simulate conversations, submit each to Vapi as a call record,
     wait for Vapi's analysisPlan to score it, blend with local scores
  3. GP-BO optimization loop: update assistant prompt, re-evaluate, repeat
  4. Validate best config on full persona set
  5. Generate before/after report, plots, and JSON results

Every conversation becomes a real Vapi call object visible in your dashboard.

Usage:
  python main.py                    # full run
  python main.py --baseline-only    # just score the baseline
  python main.py --iterations 4     # fewer iterations (faster)
  python main.py --calls-per-eval 3 # fewer calls per config
  python main.py --no-vapi          # skip Vapi submission (simulation only)
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

from agent.config import BASELINE_CONFIG, build_system_prompt, PROMPT_AXES
from agent.vapi_client import VapiClient
from evaluator.patient_simulator import simulate_conversation, PATIENT_PERSONAS
from evaluator.scorer import score_multiple_conversations, format_scores
from optimizer.gp_optimizer import GPBayesianOptimizer
from results.visualize import (
    plot_optimization_curve,
    plot_before_after,
    print_results_table,
    save_full_results,
)

console = Console()


def evaluate_config(
    config: Dict[str, int],
    n_calls: int,
    personas: List[Dict],
    vapi_client: Optional[VapiClient],
    assistant_id: Optional[str],
) -> Dict[str, float]:
    """
    Run n_calls simulated conversations for this config.
    Submits each transcript to Vapi and waits for analysisPlan scoring.
    Returns averaged scores across all conversations.
    """
    system_prompt = build_system_prompt(config)

    # Update the live Vapi assistant with this config's prompt
    if vapi_client and assistant_id:
        try:
            vapi_client.update_assistant_prompt(assistant_id, system_prompt)
        except Exception as e:
            console.print(f"  [yellow]Warning: Could not update Vapi assistant: {e}[/yellow]")

    personas_to_use = personas[:n_calls]
    results = []

    for i, persona in enumerate(personas_to_use):
        console.print(f"    [dim]Conv {i+1}/{n_calls} · persona: {persona['name']}[/dim]", end="")
        result = simulate_conversation(system_prompt, persona)
        results.append(result)
        booked = "✅" if result.appointment_booked else "❌"
        console.print(f" → booked:{booked} turns:{result.num_turns}")

    scores = score_multiple_conversations(
        results,
        vapi_client=vapi_client,
        assistant_id=assistant_id,
    )

    # Show Vapi call IDs so user can verify in dashboard
    call_ids = scores.pop("vapi_call_ids", [])
    logged = [c for c in call_ids if c]
    if logged:
        console.print(f"    [dim]Vapi calls logged: {len(logged)} — check your Vapi dashboard[/dim]")

    return scores


def run_optimization(args) -> None:
    n_calls = args.calls_per_eval
    n_iterations = args.iterations
    use_vapi = not args.no_vapi

    console.print(Panel.fit(
        "[bold cyan]Vapi Agent Optimizer[/bold cyan]\n"
        "[dim]GP Bayesian Optimization · Dental Scheduler · Vapi-backed evaluation[/dim]",
        border_style="cyan"
    ))
    console.print(f"\n[bold]Settings:[/bold] {n_iterations} iterations × {n_calls} calls each")
    console.print(f"[bold]Vapi integration:[/bold] {'enabled ✓' if use_vapi else 'disabled (--no-vapi)'}")
    console.print(f"[bold]Search space:[/bold] 81 configurations (4 axes × 3 values)\n")

    # ------------------------------------------------------------------
    # Setup: create the Vapi assistant
    # ------------------------------------------------------------------
    vapi_client = None
    assistant_id = None

    if use_vapi:
        console.rule("[bold]Setup: Creating Vapi Assistant")
        try:
            vapi_client = VapiClient()
            baseline_prompt = build_system_prompt(BASELINE_CONFIG)
            assistant_id = vapi_client.create_assistant(
                system_prompt=baseline_prompt,
                name="DentalScheduler-Optimizer",
            )
            console.print(f"[green]✓ Vapi assistant created: {assistant_id}[/green]")
            console.print(f"  View at: https://dashboard.vapi.ai/assistants/{assistant_id}")
        except Exception as e:
            # Print full response body so we can diagnose the exact Vapi error
            detail = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = f"\n  Response body: {e.response.json()}"
                except Exception:
                    detail = f"\n  Response text: {getattr(e.response, 'text', '')}"
            console.print(f"[red]✗ Vapi setup failed: {e}{detail}[/red]")
            console.print("[yellow]  Falling back to local-only mode. Check your VAPI_API_KEY.[/yellow]")
            vapi_client = None
            assistant_id = None

    eval_personas = PATIENT_PERSONAS

    # ------------------------------------------------------------------
    # Step 1: Baseline evaluation
    # ------------------------------------------------------------------
    console.rule("[bold yellow]Step 1: Baseline Evaluation")
    console.print(f"Config: {BASELINE_CONFIG}\n")

    baseline_scores = evaluate_config(
        BASELINE_CONFIG, n_calls, eval_personas, vapi_client, assistant_id
    )
    console.print(f"\n[yellow]Baseline scores:[/yellow]")
    console.print(format_scores(baseline_scores, BASELINE_CONFIG))

    if args.baseline_only:
        console.print("\n[dim]--baseline-only flag set. Done.[/dim]")
        return

    # ------------------------------------------------------------------
    # Step 2: GP-BO optimization
    # ------------------------------------------------------------------
    console.rule("[bold cyan]Step 2: GP-BO Optimization")

    optimizer = GPBayesianOptimizer(
        n_random_starts=min(3, n_iterations),
        noise=0.05,
    )

    iteration_count = [0]

    def objective(config: Dict[str, int]) -> float:
        iteration_count[0] += 1
        iter_num = iteration_count[0]
        is_random = iter_num <= optimizer.n_random_starts
        phase = "[dim]random[/dim]" if is_random else "[cyan]GP-guided[/cyan]"

        console.print(f"\n[bold]Iteration {iter_num}/{n_iterations}[/bold] ({phase})")
        console.print(f"  Config: {config}")

        scores = evaluate_config(config, n_calls, eval_personas, vapi_client, assistant_id)
        reward = scores["reward"]
        console.print(f"  → Reward: [bold green]{reward:.4f}[/bold green]")
        console.print(format_scores(scores, config))
        return reward

    best_config = optimizer.optimize(objective_fn=objective, n_iterations=n_iterations)
    optimizer.save_history("results/optimization_history.json")

    console.print(f"\n[bold green]✓ Optimization complete![/bold green]")
    console.print(f"Best config: {best_config}")
    console.print(f"Best reward: {optimizer.best_reward():.4f}")

    # ------------------------------------------------------------------
    # Step 3: Final validation on all personas
    # ------------------------------------------------------------------
    console.rule("[bold magenta]Step 3: Final Validation")
    console.print("Running full validation on best config...\n")

    final_scores = evaluate_config(
        best_config, len(eval_personas), eval_personas, vapi_client, assistant_id
    )
    console.print(f"\n[magenta]Final scores:[/magenta]")
    console.print(format_scores(final_scores, best_config))

    # ------------------------------------------------------------------
    # Step 4: Report
    # ------------------------------------------------------------------
    console.rule("[bold white]Step 4: Results")

    print_results_table(baseline_scores, final_scores, BASELINE_CONFIG, best_config)

    improvement = final_scores["reward"] - baseline_scores["reward"]
    pct = (improvement / max(baseline_scores["reward"], 0.001)) * 100
    sign = "+" if improvement >= 0 else ""
    console.print(f"\n[bold]Reward improvement: [green]{sign}{improvement:.3f} ({sign}{pct:.1f}%)[/green][/bold]")

    if assistant_id:
        console.print(f"\n[dim]Final optimized Vapi assistant:[/dim]")
        console.print(f"  https://dashboard.vapi.ai/assistants/{assistant_id}")
        console.print(f"[dim](Assistant left active — you can test it live from the Vapi dashboard)[/dim]")

    os.makedirs("results", exist_ok=True)
    plot_optimization_curve(optimizer.convergence_data())
    plot_before_after(baseline_scores, final_scores)
    save_full_results(
        baseline_scores, final_scores,
        BASELINE_CONFIG, best_config,
        [{"config": {k: int(v) if hasattr(v, "item") else v for k, v in cfg.items()}, "reward": float(r)} for cfg, r in optimizer.history],
    )

    console.print("\n[bold green]✓ All done! Check results/ for plots and JSON.[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vapi Agent Optimizer")
    parser.add_argument("--iterations", type=int, default=int(os.getenv("N_ITERATIONS", 6)))
    parser.add_argument("--calls-per-eval", type=int, default=int(os.getenv("N_CALLS_PER_EVAL", 5)))
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--no-vapi", action="store_true",
                        help="Skip Vapi API — run simulation only (no dashboard logging)")
    args = parser.parse_args()
    run_optimization(args)
