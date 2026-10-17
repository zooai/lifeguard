# Lifeguard: AI Developer & Agent Guide

This repository contains the official `zooai/lifeguard` dataset, training objectives, and diagnostic suite.

## Repository Layout
- `lifeguard/`: Python package implementing `TripartiteLifeguardLoss`, `ComplementHead`, `When2CallGate`, `FactorialANOVA`, and `LifeguardDiagnostic`.
- `datasets/`: 39,720 rule-grounded scenario rows in 6 ethical families across 4 tied views.
- `tests/`: PyTest unit tests validating numerical stability, loss convergence, and diagnostic formulas.
- `examples/`: End-to-end training and evaluation scripts.

## Key Invariants
1. Never collapse complementary probabilities: $P(\text{act}) + P(\text{stay}) = 1.0$.
2. Tripartite loss combines $\mathcal L_{\text{sup}}$, $\mathcal L_{\text{comp}}$, and $\mathcal L_{\text{mono}}$.
3. All scenarios must uphold the six core principles in `datasets/spec.json`.
4. Surface wording variance $\eta^2$ must remain under 5%.
