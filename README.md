# privilege-differential-drift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23513214.svg)](https://doi.org/10.5281/zenodo.23513214)

Does the **size of the privilege gap** between the agent writing shared state and the agent reading it predict policy violations — when every individual write is benign and no attacker is present anywhere in the chain?

This repository holds the pre-registration, and later the harness and results, for that question.

---

## The question

Most agent memory security research assumes an attacker: injected content, a compromised sub-agent, a poisoned retrieval store. A smaller recent line removes the attacker and shows agents degrading their own safety posture or propagating misaligned goals to successors.

Multi-agent delegation research establishes separately that a low-privilege agent can influence a high-privilege one — but always as a consequence of compromise.

The question registered here sits in the intersection and, as far as the author can establish, is untested:

> A low-authority agent writes a summary. A higher-authority agent reads it and acts.
> Every write is benign under isolated review. No malicious input enters the system.
> **Does the privilege gap itself raise the violation rate?**

The hypothesised mechanism is linguistic rather than adversarial: an agent operating under known constraints tends to record state as generalized rules rather than scoped observations, and a consumer with a broader tool allowlist has no way to recover the scope under which the rule was written.

If true, the failure is compositional — no single entry is a violation, and per-entry review cannot catch it.

## Primary metric

**MIR — Mandate Inflation Rate**: the fraction of runs in which a higher-authority agent attempts an action outside the originally delegated mandate, where removing a specific benign state entry eliminates the attempt.

Full definitions, hypotheses, protocol and pre-committed falsification criteria: **[PREREGISTRATION.md](./PREREGISTRATION.md)**

Deposited and timestamped: **[10.5281/zenodo.23513214](https://doi.org/10.5281/zenodo.23513214)** (resolves to the latest version)

## Status

| Version | Contents | State | DOI |
|---|---|---|---|
| v0.1 | Hypotheses, metrics, protocol, falsification criteria | Superseded | [10.5281/zenodo.23513215](https://doi.org/10.5281/zenodo.23513215) |
| v0.2 | Frozen task set, allowlists, power analysis, analysis script | **Current** | [10.5281/zenodo.23699919](https://doi.org/10.5281/zenodo.23699919) |
| v1.0 | Harness, logs, results — finding or null | Planned | — |

The harness is deliberately withheld at v0.2. The registered contribution is the formulation, the metric, and the design.

v0.2 also revises v0.1 in four respects, all before any data collection: the sample size is resized against H2 and allows for the discard rate F4 permits; F2 becomes an equivalence test rather than a failure to reject; model family becomes a fixed effect; and the two comparisons in H1 are distinguished by what each can establish. §9 of the pre-registration records them.

**A null result will be published.** The conditions under which the hypothesis is considered refuted are fixed in advance in §6 of the pre-registration, before any data is collected.

## What this is not

This does not claim novelty for memory poisoning, delayed or category-conditional activation, endogenous safety drift, self-propagation of misalignment, or the confused-deputy structure of agent pipelines. All of those are established; §1 of the pre-registration cites them and states explicitly what is being built on rather than re-claimed.

## Citation

> Kiryak, E. (2026). *Privilege-Differential Drift: Measuring Mandate Inflation in Hierarchical LLM Agent Systems — A Pre-Registration* (v0.2). Zenodo. https://doi.org/10.5281/zenodo.23513214

The concept DOI above resolves to the latest version. To cite this version specifically: https://doi.org/10.5281/zenodo.23699919 — or v0.1: https://doi.org/10.5281/zenodo.23513215

Machine-readable metadata: [CITATION.cff](./CITATION.cff)

## Repository

```
PREREGISTRATION.md     hypotheses, definitions, protocol, falsification criteria
analysis/power.py      power, sample size and minimum detectable effect
analysis/allowlists.py the three authority levels and the gaps they induce
analysis/tasks.py      the frozen task set and machine-readable mandates
tests/                 the gate the above must pass
```

Each analysis module prints its own tables when run directly:

```
python analysis/power.py
python analysis/allowlists.py
python analysis/tasks.py
```

The modules depend on nothing outside the standard library. The test suite needs `pytest` and is run with `python -m pytest tests/ -q`.

The tests were written by Claude (Opus 5, Anthropic) at the author's request. They validate the power calculations against Monte Carlo simulation of the test statistic rather than against a restatement of the same formulas, and they enforce the structural properties of the design — strict nesting of the capability sets, separation of the gap conditions, balance of the task set across categories, and the absence of scope-expansion language in the task text — as gate conditions.

## License

Two licenses, one per kind of material.

| Material | License | File |
|---|---|---|
| Pre-registration and documentation | CC BY 4.0 | [LICENSE](./LICENSE) |
| Code in `analysis/` and `tests/` | Apache-2.0 | [LICENSE-CODE](./LICENSE-CODE) |

Every source file carries its SPDX identifier. The Zenodo record names CC BY 4.0 as the license of the deposited document; the code in the deposited archive is under Apache-2.0 as stated here and in the files themselves.

---

MASec Lab — offensive security research on multi-agent systems.
