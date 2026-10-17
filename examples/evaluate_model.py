"""Example: Running Lifeguard Factorial ANOVA Diagnostics on Recorded Model Decisions.

Demonstrates evaluating a model's decisions across surface wording, track presentation order,
and consequence magnitude to detect wording dominance and contradiction failures.
"""

from lifeguard.diagnostics import LifeguardDiagnostic
from lifeguard.evaluate import format_markdown_report


def main():
    print("Generating synthetic benchmark evaluation cells...")
    # Simulate a model with high surface wording variance and polarity failure (Kai-1.5 baseline)
    synthetic_cells = []
    wordings = ["prose", "json", "yaml", "kv", "list"]
    counts = [0, 1, 2, 5, 20]

    for w in wordings:
        for order in [0, 1]:
            for n_main in counts:
                # Flawed model behavior: high wording bias, flat consequence response
                bias = 0.8 if w in ("prose", "kv") else 0.4
                p_pull = min(max(bias + 0.02 * (n_main > 0), 0.01), 0.99)
                p_stay = 0.75  # Yes-Yes contradiction bug: model says yes to stay as well!

                synthetic_cells.append({
                    "scenario": "switch" if n_main > 0 else "empty",
                    "n_main": n_main,
                    "n_side": 1,
                    "wording": w,
                    "order": order,
                    "p_pull": p_pull,
                    "p_stay": p_stay,
                    "side_is_server": False,
                })

    print(f"Evaluating {len(synthetic_cells)} cells...")
    metrics = LifeguardDiagnostic.evaluate_trolley_suite(synthetic_cells)
    report = format_markdown_report("Uncalibrated Baseline (Kai-1.5 Profile)", metrics)
    print("\n" + report)


if __name__ == "__main__":
    main()
