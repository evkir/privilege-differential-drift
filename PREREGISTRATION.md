# Privilege-Differential Drift: Measuring Mandate Inflation in Hierarchical LLM Agent Systems

**A pre-registration of hypotheses, metrics, and falsification criteria**

**Author:** Evgeny Kiryak — MASec Lab (independent)
**Version:** 0.1 (pre-registration; no results)
**Date:** 2026-10-20
**License:** CC BY 4.0
**Status:** Hypotheses and protocol registered prior to data collection. Implementation and results withheld to a later version.

---

## Abstract

Research on security failures in LLM agent memory is overwhelmingly adversary-driven: an attacker injects content, compromises a sub-agent, or poisons a retrieval store. A smaller recent line of work removes the attacker and shows that an agent can degrade its own safety posture, or propagate misaligned goals to successor agents, through ordinary memory writes. Separately, a large literature on multi-agent delegation establishes that a low-privilege agent can influence a high-privilege one — but always as a consequence of compromise or injection.

This pre-registration states a hypothesis that falls in the intersection of these lines and, to the author's knowledge, has not been tested: that the **size of the privilege gap between the agent writing shared state and the agent reading it is itself a predictor of policy violation**, under the constraint that every individual write is benign in isolation and no adversary is present at any point in the chain.

We define a primary metric — the **Mandate Inflation Rate (MIR)** — specify a three-level experimental protocol, and pre-commit to the conditions under which the hypothesis is to be considered refuted. A null result will be published.

---

## 1. Background and relation to prior work

The following summarises the state of the literature as of October 2026 and delimits what this work does **not** claim as novel.

**Adversarial memory poisoning.** A large body of work establishes that content written into persistent agent memory survives across sessions and activates later: AgentPoison; memory poisoning taxonomies and systematic studies (arXiv:2606.04329); sleeper poisoning with delayed activation (arXiv:2605.15338); environment-injected poisoning of web agents; tool hijacking via memory (MemMorph); benchmarks tracking persistence through to consequence (MemSecBench, arXiv:2607.27080). Category-conditional activation — a payload that remains dormant until a specific class of data is processed — is demonstrated by Trojan Hippo. **None of this is claimed as novel here.**

**Endogenous degradation without an adversary.** Misevolution (Shao et al., arXiv:2509.26354) shows that mere accumulation of task experience degrades safety alignment, with safety-relevant memories diluted or displaced by performance-oriented ones. Survey work on self-evolving systems (arXiv:2606.23075) names this *endogenous safety drift*. Self-Propagating Misalignment (arXiv:2610.04083) removes the adversary entirely: an agent creates misaligned memory content that successor agents inherit, persisting across 100 unrelated sessions and transferring to more capable and more safety-aligned successors. **This work explicitly builds on, and does not re-claim, these results.**

**Delegation and trust boundaries.** Multi-agent delegation threat models (arXiv:2609.00267), privilege escalation taxonomies in agent systems (arXiv:2601.11893), trust propagation and structural containment (arXiv:2609.17648), and bounded-authority / capability-envelope proposals (arXiv:2608.15888) all establish the confused-deputy structure of agent pipelines: a low-privilege agent's output can be accepted as authority by a high-privilege one. In every case the triggering cause is modelled as compromise, injection, or an untrusted peer. **The causal premise, not the structure, is what differs here.**

### 1.1 Gap statement

The literature treats the privilege boundary as a *channel* through which an externally-originated compromise travels. It does not treat the privilege **differential** as an independent variable that may itself generate violations in the absence of any compromise.

The specific question registered here:

> Given a shared state artifact written by an agent at authority level *i* and consumed by an agent at authority level *j* > *i*, where every individual write is benign under isolated review and no adversarial input enters the system at any point — does the magnitude of the gap between *i* and *j* predict the rate at which the consuming agent attempts actions outside the originally delegated mandate?

### 1.2 Proposed mechanism

The hypothesised mechanism is linguistic, not adversarial. An agent operating under known constraints has an incentive to record state in a form that survives its own limitations — as generalized procedural rules ("X is approved", "the standard approach is Y") rather than as scoped observations ("under task T, with permissions P, X was observed"). This is a reasonable and often desirable summarisation behaviour. A consumer at a higher authority level, holding a broader tool allowlist, has no mechanism to recover the scope under which the rule was written, and may read a rule as a mandate.

If true, this predicts that the failure is **compositional**: no single write is a violation, no single read is a violation, and the violation exists only in the sequence — which has direct consequences for any defence operating at the level of individual-entry review.

---

## 2. Definitions

Let an agent system consist of agents **A₀ … Aₙ**, each assigned an authority level with an associated capability set.

**D1 — Capability set.** *C(Aᵢ)* is the set of tool invocations permitted to agent *Aᵢ*, enforced deterministically at the tool boundary by a policy gate, not by prompt instruction.

**D2 — Authority ordering.** Levels are constructed such that *C(A₀) ⊂ C(A₁) ⊂ C(A₂)*. The ordering is strict and by construction, not inferred.

**D3 — Privilege gap.** For a write by *Aᵢ* consumed by *Aⱼ*, the gap is
  **Δ(i→j) = |C(Aⱼ) \ C(Aᵢ)|**
normalised as **Δ̂ = Δ(i→j) / |C(Aⱼ)|** where cross-condition comparison requires it.

**D4 — Mandate.** *M* is the task scope assigned to the originating agent by the human principal, expressed as the set of actions *allowed(M)* that a correct execution may contain. *M* is fixed at task start and is not rewritten by any agent.

**D5 — Shared state.** *S* is the set of entries written by agents during execution and made available to subsequent agents — summaries, memory entries, notes, handoff artifacts. *S* is append-only and hash-chained.

**D6 — Atomic benignness.** An entry *e ∈ S* is *atomically benign* if an independent reviewer, shown *e* in isolation without downstream context, judges it a legitimate product of the writing agent's assigned task. A run in which any entry fails this test is **discarded, not counted as a positive**. This constraint is what distinguishes the phenomenon from self-propagating misalignment, where the originating agent is already misaligned.

**D7 — Mandate inflation event.** An attempted invocation *a* by consuming agent *Aⱼ* such that:
  (i) *a ∉ allowed(M)* — the action falls outside the original mandate;
  (ii) *a ∈ C(Aⱼ)* — the action is within the consumer's own capability set, so this is not an escalation attempt but an authorised action taken for an unauthorised purpose;
  (iii) there exists a non-empty *E ⊆ S* such that re-running *Aⱼ* with *S \ E* under a fixed seed does not produce *a* — the state, not the task, is causal;
  (iv) every *e ∈ E* is atomically benign per D6.

All four conditions are required. Condition (iii) is what makes the claim causal rather than correlational.

---

## 3. Metrics

**M1 — Mandate Inflation Rate (primary).**
  **MIR(i→j) = (runs containing ≥1 mandate inflation event) / (total valid runs)**
A run is *valid* if it completes and no entry in *S* fails atomic benignness.

**M2 — Attribution Identification Rate (secondary).**
  **AIR = (inflation events for which a unique minimal causal set E is recovered) / (total inflation events)**
Minimality: no proper subset of *E* satisfies D7(iii).

**M3 — Per-category MIR (secondary).** MIR computed within each data category, including the control category.

**M4 — Inflation depth (exploratory).** Number of intervening agent hops between the write of the earliest entry in *E* and the inflation event.

Detection of D7(i)–(ii) is performed by the deterministic policy gate. No LLM judge is used for the primary metric. An LLM judge is used only for atomic benignness (D6), where it is a *discard* criterion — i.e. it can only reduce the measured MIR, never inflate it. This asymmetry is deliberate.

---

## 4. Hypotheses

**H1 (primary).** MIR increases monotonically with Δ̂. Specifically, MIR(0→2) > MIR(0→1) and MIR(0→2) > MIR(1→2), with the difference significant at α = 0.05 after correction for multiple comparisons.

**H2.** MIR differs significantly across data categories, with at least one non-control category exceeding the control category. Tested on per-category MIR pooled across model families, at the sample size registered in §5; per-cell MIR is descriptive only.

**H3.** A unique minimal causal entry set is recoverable for the majority of inflation events: AIR ≥ 0.70.

**H4 (exploratory, not confirmatory).** Inflation events occur at depth > 1 — i.e. the causal entry was written more than one hop upstream of the violation.

---

## 5. Protocol

**Authority levels.** Three levels with strictly nested tool allowlists: L0 read-only; L1 read plus bounded local write; L2 read, write, and egress-capable tools. Exact allowlists fixed before data collection and published with results.

**Data categories.** Four substantive categories — financial, personal data, infrastructure/credentials, source code — plus one control category selected to be semantically neutral. Category choice is motivated by prior evidence of category-conditional activation in the adversarial setting; the question here is whether the same conditioning appears without a planted trigger.

**Tasks.** All tasks are benign, realistic, and individually completable within the assigned mandate. No task contains instructions, hints, or content designed to induce scope expansion. Task set is authored before the hypotheses are tested and is frozen.

**Conditions.** 3 gap conditions × 5 categories = 15 cells.

**Sample size.** Three gap conditions × five data categories × ≥ 3 model families from distinct providers. The power analysis is complete and frozen with this version, assuming a baseline MIR of 0.10 and a minimum effect of interest of 0.10 absolute difference.

The binding constraint is H2, not H1. The per-category comparisons are made at a Holm planning α of 0.0125 (four substantive categories against the control), where 80% power on the registered effect of interest requires **283 valid runs per category arm** — 95 valid runs per cell once pooled across three model families. H1 is satisfied by any design that satisfies H2: pooled across categories and families its power exceeds 0.99 and it resolves differences down to 0.036 absolute.

Because F4 tolerates a discard rate of up to 30% under atomic benignness (D6), the collection target is set above the valid-run requirement: **135 runs per cell**, i.e. 3 × 5 × 3 × 135 = **6,075 runs total**. This leaves ≥ 95 valid runs per cell at the worst discard rate F4 permits. At a 15% discard rate 111 runs per cell would suffice; the higher target is registered so the design does not require resizing if discard runs high.

Per-cell MIR is reported descriptively only. At 100 runs in a single cell the smallest detectable effect is 0.185 absolute — nearly twice the registered minimum effect of interest — so no confirmatory claim is made at the level of an individual cell.

Power, sample size and minimum-detectable-effect figures are produced by `analysis/power.py`, frozen with this version. The accompanying test suite validates them against Monte Carlo simulation of the test statistic rather than against a restatement of the same formulas.

**Statistical plan.** Mixed-effects logistic regression with inflation as the outcome, Δ̂ and category as fixed effects, and task identifier as a random effect. Model family enters as a **fixed** effect with three levels: three groups are too few to estimate a variance component stably, and the registered design treats the three providers as the population of interest rather than a sample drawn from one. Holm correction across the pre-specified comparisons, with the design powered at the most stringent step of the sequence (α/k) rather than the least. Bootstrap confidence intervals on per-cell MIR. The analysis script is written and frozen before data collection.

**Logging.** Append-only, hash-chained event log capturing every state write, every state read, every tool invocation and every gate decision, with run seed. Logs are published with results.

**Attribution procedure.** Minimal causal set recovery proceeds by systematic ablation of state entries with re-execution under fixed seed. The specific search strategy and its optimisations are withheld from this version and will be published together with results.

---

## 6. Pre-committed falsification criteria

The hypothesis is to be considered **refuted**, and the null result published, if:

**F1.** MIR shows no significant dependence on Δ̂ at α = 0.05 under the pre-specified model, across at least 3 model families at the registered sample size. *(Refutes H1 — the core claim. If F1 triggers, the work is a null result on privilege differential, not a finding about agent memory.)*

**F2.** The control category is shown to be statistically **equivalent** to every substantive category, by two one-sided tests against an equivalence margin of 0.10 absolute MIR at α = 0.0125 per one-sided test, on per-category MIR pooled across model families. *(Refutes H2 — the observed behaviour is unconditioned and reduces to known misevolution.)*

A failure to reject a difference does not trigger F2. At any realistic sample size that outcome is consistent both with equivalence and with an effect the design did not detect, so stating the criterion as a non-significant difference would make F2 fire on insufficient data rather than on evidence of no conditioning. The registered margin of 0.10 is the minimum effect of interest: the claim being pre-committed is that any categorical conditioning is smaller than the effect this work set out to find. At the registered sample size this procedure has 0.93 power when the true difference is zero.

**F3.** AIR < 0.50. *(Refutes H3 — the phenomenon exists but is not forensically attributable, which removes the defensive contribution.)*

**F4.** Discard rate under atomic benignness (D6) exceeds 30% of runs. *(The premise fails: the originating agents are not behaving benignly, and the result collapses into the self-propagating-misalignment setting already established in arXiv:2610.04083.)*

F1 and F4 are the decisive ones. F4 in particular is the criterion that keeps this work distinct from prior results rather than a restatement of them.

---

## 7. Threats to validity

Recorded in advance, without mitigation claims:

- **Author bias in scenario design.** The same author writes the tasks and the expected-action sets. Partial mitigation: tasks frozen before hypotheses are tested; full task set published. Not fully mitigable by a single author.
- **Benignness judgment is subjective.** D6 depends on an LLM judge plus spot human review. Inter-rater agreement to be reported.
- **Non-determinism.** Fixed seeds do not guarantee reproducibility across providers. Ablation results are therefore reported with repetition counts, not as single-shot counterfactuals.
- **Mandate specification.** *allowed(M)* must be enumerable to be enforceable, which biases toward tasks with crisp action boundaries and may not generalise to open-ended work.
- **Construct validity of Δ̂.** Counting allowlist entries treats all capabilities as equally weighted, which they are not. Alternative weightings to be reported as a sensitivity analysis.
- **Model version drift.** Provider models change under fixed names. Collection dates and version identifiers logged per run.

---

## 8. Scope and ethics

All work is conducted in a local sandbox with synthetic data. No production system, third-party service, or live credential is involved at any point. There is no adversary in the threat model and no attack payload is developed, so no exploit artifact is produced or withheld.

Should the hypothesis be confirmed, the finding describes a property of a common architectural pattern rather than a vulnerability in a specific product. No coordinated disclosure to a single vendor is therefore anticipated; if a specific framework is found to be disproportionately affected, its maintainers will be contacted before results are published.

---

## 9. Versioning and priority

This document records the formulation, metric definitions, and falsification criteria as of the stated date. It contains no results.

- **v0.1 (this version)** — hypotheses, metrics, protocol, falsification criteria.
- **v0.2** — frozen task set, allowlists, power analysis, analysis script.
- **v1.0** — harness implementation, full logs, results, and either the finding or the null.

Implementation details are deliberately withheld at v0.1. The registered contribution is the formulation and the metric, not the code.

---

## 10. How to cite

> Kiryak, E. (2026). *Privilege-Differential Drift: Measuring Mandate Inflation in Hierarchical LLM Agent Systems — A Pre-Registration*. MASec Lab. Zenodo. DOI: [10.5281/zenodo.23513214](https://doi.org/10.5281/zenodo.23513214)

The concept DOI above always resolves to the latest version. Version-specific DOIs: v0.1 — [10.5281/zenodo.23513215](https://doi.org/10.5281/zenodo.23513215).

---

## 11. References

Referenced by identifier; full bibliography to accompany v1.0.

1. arXiv:2610.04083 — Self-Propagating Misalignment in LLM Agents, and Why Auditing or Disabling Memory Is Not Enough
2. arXiv:2509.26354 — Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents
3. arXiv:2606.23075 — Safety in Self-Evolving LLM Agent Systems: Threats, Amplification, and Case Studies
4. arXiv:2606.04329 — From Untrusted Input to Trusted Memory: A Systematic Study of Memory Poisoning Attacks in LLM Agents
5. arXiv:2607.27080 — MemSecBench: Tracking Agent Memory Poisoning from Persistence to Consequence and Repair
6. arXiv:2605.15338 — Hidden in Memory: Sleeper Memory Poisoning in LLM Agents
7. arXiv:2608.01759 — Benign Alone, Harmful Together: Exploiting Experience Composition in Self-Evolving LLM Agents
8. arXiv:2609.00267 — Delegation Without Trust: An Empirical Gap Analysis
9. arXiv:2609.17648 — Trust Propagation and Structural Containment in Multi-agent LLM Pipelines
10. arXiv:2601.11893 — Taming Various Privilege Escalation in LLM-Based Agent Systems
11. arXiv:2608.15888 — Bounded Agents: Delegation Security for Multi-Agent AI Systems
12. arXiv:2609.28586 — Agent Approval Laundering: Transitive Effects Beyond the Approved Invocation
13. Hardy, N. (1988). The Confused Deputy. ACM SIGOPS Operating Systems Review, 22(4).
