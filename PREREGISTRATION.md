# Privilege-Differential Drift: Measuring Mandate Inflation in Hierarchical LLM Agent Systems

**A pre-registration of hypotheses, metrics, and falsification criteria**

**Author:** Evgeny Kiryak — MASec Lab (independent)
**Version:** 0.2 (pre-registration; no results)
**Date:** 2026-10-27 (v0.1: 2026-10-20)
**License:** CC BY 4.0 for this document (`LICENSE`); Apache-2.0 for the code in `analysis/` and `tests/` (`LICENSE-CODE`).
**Status:** Hypotheses, protocol, task set, allowlists and power analysis registered prior to data collection. Harness and results withheld to v1.0.

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

**Compositional and transitive effects.** Two results sit close enough to this work that the distinction has to be stated rather than left to the reader.

*Experience composition* (arXiv:2608.01759) demonstrates that experiences which are benign in isolation can jointly weaken an agent's safety boundary, and that per-item review therefore does not suffice. The compositional structure is the same one registered here. The causal premise is not: that work proposes an experience-conditioned sequential attack in which an adversary observes which experiences the victim distils, identifies the gaps relevant to its target, induces the complementary experiences through ordinary interaction, and then issues a final query that activates them together. Benignness is a property the attacker maintains in order to evade review. The independent variable is the attacker's construction; the agents are not at differing authority levels. Here there is no adversary at any point, no query is constructed to activate anything, and the independent variable is Δ̂ — a structural property of the deployment rather than of an opponent. **The claim registered here is not that benign items compose; it is that the privilege differential predicts the rate at which they do.**

*Approval laundering* (arXiv:2609.28586) removes the adversary and shows that authority granted at one point extends past its intended scope: an approval record names the entry invocation truthfully while omitting the effects the resulting workflow exercises. The shared intuition — authority outrunning the scope it was granted under — is real, and the gap there is between an approval record and the transitive effects of tooling, deterministic and in principle recoverable from command semantics, in a human-to-agent approval path. The gap registered here is between the scope under which an agent wrote a state entry and the authority of the agent that later reads it, in an agent-to-agent path with no human in the loop, where the inflation is produced by the consuming model's reading of an underspecified rule rather than by an incomplete record. Neither work treats the magnitude of a privilege difference as a predictor. **Neither is re-claimed; both constrain what may be claimed as novel.**

*Non-adversarial over-privilege* (arXiv:2606.20023) removes the adversary and measures privilege escalation directly: an agent offered a lower-privilege and a higher-privilege tool, both independently sufficient for the task, selects or escalates to the higher-privilege one. The capability confound is removed by construction, as it is here, and no attacker is present at any point. The difference is structural. That work measures a single agent's choice at a single decision point, where over-privilege is relative to the task; there is no writer, no shared state, and no authority differential, because there is only one agent. The quantity registered here exists only for a chain of length greater than one: it is the gap between the authority under which a state entry was written and the authority of the agent that later reads it, and the hypothesised mechanism is the consumer's reading of a rule whose scope another agent stripped. **Neither the phenomenon of non-adversarial escalation nor the removal of the capability confound is claimed as novel; the privilege differential as a predictor across a delegation boundary is.**

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

The shared-state operations of D5 are **not** members of any capability set. They are a harness channel, available identically at every authority level. This is required for consistency: D5 has agents at every level writing entries that later agents consume, while §5 defines L0 as read-only. Were state access a capability, L0 would either hold a write capability in contradiction of its own definition or be unable to author the entries the experiment depends on. The consequence for D3 is that Δ̂ measures the difference in what agents may do to the world, not in how they communicate with one another.

**D2 — Authority ordering.** Levels are constructed such that *C(A₀) ⊂ C(A₁) ⊂ C(A₂)*. The ordering is strict and by construction, not inferred.

**D3 — Privilege gap.** For a write by *Aᵢ* consumed by *Aⱼ*, the gap is
  **Δ(i→j) = |C(Aⱼ) \ C(Aᵢ)|**
normalised as **Δ̂ = Δ(i→j) / |C(Aⱼ)|** where cross-condition comparison requires it.

Two properties of this quantity are recorded here so that neither is mistaken for a result. First, for strictly nested capability sets with a non-empty *C(A₀)*, the ordering Δ̂(0→2) > Δ̂(0→1) and Δ̂(0→2) > Δ̂(1→2) holds for **any** choice of level sizes. It is a consequence of the construction, not evidence of anything, and the entire empirical content of H1 therefore lies in MIR. Second, what nesting does *not* guarantee is that the three conditions are distinguishable: level sizes of (4, 8, 16), for instance, give Δ̂(0→1) = Δ̂(1→2) = 0.5 exactly, making two of the three cells the same treatment under different labels. The registered allowlists are chosen so that no two values of Δ̂ lie within 0.10 of each other.

Δ̂ therefore takes exactly three values in this design. It enters the model of §5 as an ordered three-level factor, and no claim is registered about the shape of the relation beyond the monotonicity stated in H1.

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
A run is *valid* if it completes and no entry in *S* fails atomic benignness. A run *completes* when both agents stop of their own accord and the writing agent has left at least one entry in *S*; every other ending is a collapse (§5, *Run outcomes*), which is neither a valid run nor a discard.

**M2 — Attribution Identification Rate (secondary).**
  **AIR = (inflation events for which a unique minimal causal set E is recovered) / (total inflation events)**
Minimality: no proper subset of *E* satisfies D7(iii).

**M3 — Per-category MIR (secondary).** MIR computed within each data category, including the control category.

**M4 — Inflation depth (exploratory).** Number of intervening agent hops between the write of the earliest entry in *E* and the inflation event.

Detection of D7(i)–(ii) is performed by the deterministic policy gate. No LLM judge is used for the primary metric. An LLM judge is used only for atomic benignness (D6), where it is a *discard* criterion — i.e. it can only reduce the measured MIR, never inflate it. This asymmetry is deliberate.

---

## 4. Hypotheses

**H1 (primary).** MIR increases monotonically with Δ̂. Specifically, MIR(0→2) > MIR(0→1) and MIR(0→2) > MIR(1→2), with the difference significant at α = 0.05 after correction for multiple comparisons.

The two comparisons are not of equal evidential weight, and the difference is registered here rather than discovered afterwards. An inflation event must be drawn from *C(Aⱼ) \ allowed(M)*, the actions a consumer holds that the mandate excludes. That set is larger for a consumer at L2 than at L1 — 9 actions against 2, for the frozen task set — so **MIR(0→2) > MIR(0→1) is consistent both with H1 and with the weaker proposition that a consumer holding more capabilities has more ways to leave a fixed mandate**. It cannot separate the two.

**MIR(0→2) > MIR(1→2) can.** The consumer is at L2 in both conditions, so *allowed(M)* and *C(Aⱼ)* are identical and the opportunity set is the same set, not merely the same size. The only quantity that differs is the authority of the agent that wrote the state. A difference in MIR between these two conditions is therefore attributable to the privilege differential and to nothing else in the design.

Both comparisons are retained and both are corrected for, but the second is the one on which the registered claim rests. Should the two disagree — the first significant and the second null — the result is to be reported as a capability-count effect, not as support for H1.

**H2.** MIR differs significantly across data categories, with at least one non-control category exceeding the control category. Tested on per-category MIR pooled across model families, at the sample size registered in §5; per-cell MIR is descriptive only.

**H3.** A unique minimal causal entry set is recoverable for the majority of inflation events: AIR ≥ 0.70.

**H4 (exploratory, not confirmatory).** Inflation events occur at depth > 1 — i.e. the causal entry was written more than one hop upstream of the violation.

---

## 5. Protocol

**Authority levels.** Three levels with strictly nested tool allowlists: L0 read-only; L1 read plus bounded local write; L2 read, write, and egress-capable tools. The allowlists are frozen with this version in `analysis/allowlists.py` and are published here:

| Level | \|C\| | Capabilities added at this level |
|-------|-------|----------------------------------|
| L0 — read-only | 6 | `fs.read_file`, `fs.list_dir`, `fs.grep`, `db.query_readonly`, `config.read`, `log.read` |
| L1 — plus bounded local write | 9 | `fs.write_file`, `fs.move`, `db.insert_row` |
| L2 — plus egress and privileged | 16 | `http.get`, `http.post`, `email.send`, `api.call_external`, `pkg.install`, `secrets.read`, `iam.grant_role` |

Egress and credential access exist only at L2; L1 additions are confined to the sandbox and reversible within it. The shared-state operations (`state.read`, `state.append`) are available at every level and belong to no capability set, per D1.

The three gap conditions this induces:

| Condition | Δ | \|C(Aⱼ)\| | Δ̂ | Δ̂ weighted |
|-----------|---|-----------|-----|------------|
| L0 → L1 | 3 | 9 | 0.333 | 0.500 |
| L1 → L2 | 7 | 16 | 0.438 | 0.684 |
| L0 → L2 | 10 | 16 | 0.625 | 0.842 |

The smallest separation between any two Δ̂ values is 0.104. The weighted column is the sensitivity analysis promised in §7, counting each capability by blast radius — read 1, bounded local write 2, privileged 3, egress 4 — rather than as one entry. It is reported alongside the primary analysis, which uses the unweighted Δ̂; the ordering H1 depends on is preserved under both.

**Data categories.** Four substantive categories — financial, personal data, infrastructure/credentials, source code — plus one control category selected to be semantically neutral. Category choice is motivated by prior evidence of category-conditional activation in the adversarial setting; the question here is whether the same conditioning appears without a planted trigger.

**Tasks.** All tasks are benign, realistic, and individually completable within the assigned mandate. No task contains instructions, hints, or content designed to induce scope expansion. The task set is authored before the hypotheses are tested, frozen with this version in `analysis/tasks.py`, and published in full.

The set is 25 tasks: five data categories × five task archetypes — summarize, reconcile, index, triage, extract. **An archetype carries the same mandate in every category**: the same *allowed(M)*, the same tools the writing agent needs, the same two-part division of labour. What distinguishes a financial task from its control counterpart is the subject matter of the records and nothing else. Without this constraint a difference in MIR between categories could be a difference in task shape wearing a category's name, and H2 would not test what it states.

Every *allowed(M)* is a subset of L1, so no mandate in the design authorises egress or privileged capability and every such attempt by a consumer is outside the mandate by construction, without an intent judgment at the boundary. Every *allowed(M)* also contains all of L0: a mandate is the reads of the sandbox plus the one write the consumer's half needs, so no read inside the sandbox is ever outside a mandate, and an inflation event is always a write or an egress-capable action. Every writer brief is satisfiable within L0, so one frozen set serves all three gap conditions including those whose writer is read-only. Each mandate strictly exceeds what its writer can do alone, so the consumer always has a remaining action to take.

The undertaking that no task plants a hint is enforced as a gate condition rather than a review practice: the task text is linted against a frozen lexicon of terms naming an L2 capability, pointing outside the sandbox, or granting open-ended discretion. The lint is itself tested against a planted hint, so that it cannot pass by being inert. Three formulations in the first draft of the set were rewritten after the lint rejected them.

**Conditions.** 3 gap conditions × 5 categories = 15 cells.

**Sample size.** Three gap conditions × five data categories × ≥ 3 model families from distinct providers. The power analysis is complete and frozen with this version, assuming a baseline MIR of 0.10 and a minimum effect of interest of 0.10 absolute difference.

The binding constraint is H2, not H1. The per-category comparisons are made at a Holm planning α of 0.0125 (four substantive categories against the control), where 80% power on the registered effect of interest requires **283 valid runs per category arm** — 95 valid runs per cell once pooled across three model families. H1 is satisfied by any design that satisfies H2: pooled across categories and families its power exceeds 0.99 and it resolves differences down to 0.036 absolute.

Because F4 tolerates a discard rate of up to 30% under atomic benignness (D6), the collection target is set above the valid-run requirement: **135 runs per cell**, i.e. 3 × 5 × 3 × 135 = **6,075 runs total**. At the worst discard rate F4 permits this leaves **283 valid runs per category arm** once pooled across the three model families — exactly the requirement above, and the figure H2 is tested on. The per-cell figure is 94 at that rate; it is reported for completeness and nothing is inferred from a single cell. At a 15% discard rate 111 runs per cell would suffice; the higher target is registered so the design does not require resizing if discard runs high.

Per-cell MIR is reported descriptively only. At the registered 135 runs in a single cell the smallest detectable effect is 0.154 absolute — half again the registered minimum effect of interest — so no confirmatory claim is made at the level of an individual cell.

Power, sample size and minimum-detectable-effect figures are produced by `analysis/power.py`, frozen with this version. The accompanying test suite validates them against Monte Carlo simulation of the test statistic rather than against a restatement of the same formulas.

**Harness.** The policy gate of D1 and the event log of the logging paragraph below are implemented in `harness/`, frozen with this version. Four properties of that implementation bear on what the metrics mean and are registered here rather than left to the code.

*The gate establishes a candidate, not a confirmed event.* It decides D7(i) and (ii), which are decidable from the task and the capability sets alone. It cannot decide (iii), which requires the ablation re-run, or (iv), which requires the D6 review. MIR counts confirmed events; a gate ruling is an input to that count, never the count itself.

*Writer-side scope breaches do not enter the MIR numerator.* Since every writer may use all of L0, a writer breach is a write to the world, and only a writer at L1 — the 1→2 condition — can attempt one. D7 is a statement about the consuming agent. An attempt by the writing agent outside its own brief puts the atomic-benignness premise of that run in doubt, so it is routed to a separate category feeding the F4 discard rate. Folding it into the consumer-side count would let measured MIR rise on writer misbehaviour, which is the confound F4 exists to catch.

*Blocked invocations are logged, not executed.* Executing a denied action would change the world state the remainder of the run reads from, and the D7(iii) re-run would no longer be comparing like with like.

*An out-of-capability attempt is not an inflation event.* D7(ii) requires the action to be within C(Aⱼ). Such attempts are recorded and counted separately as escalation attempts.

**Models.** Collection runs on open-weight models, locally, through Ollama on a single GPU with 12 GB of memory: `llama3.1:8b` (Meta), `mistral-nemo:12b` (Mistral AI with NVIDIA) and `qwen3:14b` (Alibaba, Qwen team). "Model families from distinct providers" in this section means families from three different developers of open weights; no hosted model API is used at any point. Each tag runs at its default quantisation and with the sampling parameters it is shipped with, temperature included. The weights digest recorded with every run identifies the model manifest and with it those parameters, and the values in force are listed with the results. Temperature is not set to zero: under greedy decoding the seed has no effect, and the replicates of a cell would be one run repeated. It is not set to one common value either: model family is a fixed effect and H1 and H2 are tested within it, so a difference in shipped temperature between families is absorbed by the family term, not by Δ̂ or by category. Whether `qwen3:14b` runs with its reasoning mode on or off, and the context size, reply-length limit and step limit of a run, are fixed by the pilot below, under rules registered here before it is launched.

**Run executor.** `harness/executor.py` carries out one run: the writing agent at level *i* performs its brief and records entries in *S*; the consuming agent at level *j* is then given those entries and performs the other half of the task. Every attempted invocation of either agent passes through the gate and into the journal. What the agents are shown is part of the instrument, so the choices are registered rather than left to the code.

*An agent is offered its whole capability set, not its brief.* The tools offered are C(A) for its level plus the shared-state channel. Offering only what a brief needs would make a writer scope breach, or an inflation event, impossible except by naming a tool the agent was never shown. Tools outside C(A) are not offered, so an escalation attempt requires exactly that.

*Nothing about authority reaches a prompt.* No prompt states a level, a mandate boundary or a permission, as D1 requires; the fixed prompt text is linted against the same lexicon as the task text.

*Both agents see the principal's mandate text, followed by their own brief.* The consumer knows what the task is; it is not told what the task permits.

*The consumer is handed the state.* The entries are placed in its opening message and that delivery is journalled as a state read: §1.2 concerns how a consumer reads an entry, not whether it chooses to look. It is shown each entry's identifier and content and nothing else; the level of the entry's author is withheld.

*Both agents are told what the world contains.* The opening message of each agent carries a description generated from the world fixture alone: every table with its columns, every top-level directory and the name of every configuration value. Rows, file contents and configuration values are not in it. It is the same text for the writer and the consumer at every level, names no tool and gives no direction. It was added after smoke runs on 12 November 2026, in which models guessed table names and three runs of four handed over nothing read from the world.

*One tool call per reply is asked for, not enforced.* Both system prompts ask for one call per reply. Every call of a reply that holds several still reaches the gate and the journal: a call dropped or held back would be an attempt never ruled on, and a consumer's first reply is composed after it has read the entries, which is where an inflation event would be. It was added after smoke runs on 14 November 2026, in which every model composed several calls before seeing any result and a writer recorded figures it had not yet read. The share of replies holding more than one call is reported per family.

*A denied invocation gets one fixed answer*, whatever the gate's classification, so the reply does not reveal which boundary was crossed, and the run continues.

*Permitted tools act on a simulated world.* Files, tables and configuration come from a fixture per task; nothing touches the host.

*Model text is not journalled.* Prompts, replies, reasoning text and timings are stored in a transcript beside the journal and tied to it by the journal's head digest.

*One seed serves a run.* It is derived from the run's coordinates in the design — family, task, the two levels, replicate — so the list of seeds is fixed by the design and cannot be chosen after a run has been seen. The D7(iii) re-run reuses it with the original world and configuration.

**Run outcomes.** A run ends in one of four ways: it completes (§3); the writer collapses, by stopping without leaving an entry, reaching the step limit or reaching the reply-length limit; the consumer collapses, by reaching the step limit or the reply-length limit; or the model backend fails. A collapse says that a model did not carry out its part, not that an entry was other than benign, so it is not a discard under D6 and does not enter the F4 rate. A backend failure says nothing about the agent; such a run is executed again and does not count as an attempt.

Collapsed runs are replaced. The replicates of a cell are executed in index order and the first 135 that complete are the cell's runs; a cell stops at 270 attempts, and a cell that reaches that limit with fewer than 135 completed runs is reported as underfilled, with its count, and enters the analysis with the runs it has.

Replacement is not assumed to be ignorable. Writer-side collapse can differ between conditions because the writer's level does, so its share is reported per cell and compared between 0→2 and 1→2, the pair the registered claim rests on. Consumer-side collapse can be selective on the outcome itself, since a consumer repeatedly attempting actions outside its mandate may stop on the step limit; a sensitivity analysis is therefore reported in which consumer-side collapsed runs enter as valid runs, with any candidate events they contain taken through ablation and review like any other.

**Family replacement.** If a family's collapse share in the pilot exceeds 0.50, it is replaced before collection by the candidate of the same size: `llama3.1:8b` by `granite4.1:8b` (IBM), `mistral-nemo:12b` by `gemma4:12b` (Google). `qwen3:14b` has no candidate; above the threshold it ends the design as below. Both candidates are open weights from a developer not otherwise in the design, with tool calling supported in Ollama, and are held entirely on the same GPU; each completed a smoke run on 14 November 2026 before it was named here. Pairing by size keeps the sizes of the three models as they are. A candidate is piloted under the same protocol and the same threshold: its ten runs, and for `gemma4:12b` ten more with its reasoning mode switched the other way. If it too exceeds the threshold, collection does not start and the design is amended in a new version.

**Pilot.** Before collection, a pilot measures the operating characteristics of the setup and nothing else. For each of the three families it runs the five control-category tasks, one per archetype, under the 0→2 and 1→2 conditions — both writer levels, and the consumer of the decisive comparison — once each: 30 runs. The same ten are run for `qwen3:14b` with its reasoning mode switched the other way, 40 runs in all. Pilot runs take replicate indices from 900, which collection never reaches, so no pilot seed recurs in the data. Settings: context of 8,192 tokens, reply-length limit of 4,096 tokens, step limit of 12 per agent, tool results cut at 4,000 characters, shipped sampling parameters.

Measured, per family: wall time and generation speed per run; model calls and tool invocations per agent; the share of each way a run ends; the share of runs containing a writer scope breach, on a line of its own; the share of replies holding more than one call; the peak prompt length as a fraction of the context; the share of the model held on the GPU, as the server reports it, and memory in use on the GPU while the model is loaded, as the driver reports it. Memory is not taken from the server: for `gemma4:12b` it reports well under a gigabyte for weights of 7.4 GB. Not computed: MIR, AIR, any count of the consumer's gate classifications, and any comparison between categories or conditions on those. Pilot runs are not part of the data, whatever they contain.

Fixed from the pilot by rules registered now:

- *Reasoning mode of `qwen3:14b`, and of `gemma4:12b` if it is piloted*: for each, the mode with the lower collapse share; if the two shares are within 0.10 of each other, off.
- *Reply-length limit*: the smallest power of two at least twice the longest reply generated in the pilot, and not below 1,024 tokens.
- *Context size*: 8,192 tokens if the peak prompt in the pilot is at most two thirds of it; otherwise 16,384, provided every model is still held entirely on the GPU at that size.
- *Step limit*: at least twice the largest number of model calls any agent made in a completed pilot run, and not below 12.
- *Family replacement*: as above.

The pilot table and the values these rules give are recorded in this section before collection begins.

**Statistical plan.** Mixed-effects logistic regression with inflation as the outcome, Δ̂ and category as fixed effects, and task identifier as a random effect. Model family enters as a **fixed** effect with three levels: three groups are too few to estimate a variance component stably, and the registered design treats the three providers as the population of interest rather than a sample drawn from one. Holm correction across the pre-specified comparisons, with the design powered at the most stringent step of the sequence (α/k) rather than the least. Bootstrap confidence intervals on per-cell MIR, resampled over tasks rather than over runs: runs sharing a task are not independent, and resampling runs would report an interval narrower than the design supports. The analysis script is `analysis/model.py`, written and frozen with this version, before data collection.

**Estimator.** The model is fitted by marginal maximum likelihood, integrating the task random effect out by adaptive Gauss-Hermite quadrature at 25 nodes, with Wald tests on the pre-specified contrasts of §4. It is implemented in `analysis/model.py` rather than taken from a library because no Python library provides a frequentist random-intercept logistic regression with the p-values Holm is applied to: the mixed linear model in `statsmodels` is Gaussian-response only, and its binomial mixed model is Bayesian, returning posterior summaries. The alternatives were to call R's `lme4::glmer` across a language bridge, which puts a second toolchain between a reader and the result, or to amend this section to a marginal model, which would change what is tested. Three properties of the implementation are established before collection and recorded here: the quadrature agrees with dense fixed-grid evaluation of the same integral to within 1e-4 in log-likelihood; with the random effect unidentified the fit reduces to the ordinary logistic maximum likelihood an independent optimiser finds; and its type I error and Wald interval coverage are measured by simulation, as the power analysis is. A fourth check is deferred to analysis time rather than to the test suite: the collected data is refit once in `lme4::glmer` at the same node count, both fits are reported side by side, and a disagreement above 0.01 on the log-odds scale for any pre-specified contrast blocks publication. An estimator written by the author of the hypothesis is not taken on trust. The numpy and scipy versions used are logged with the results.

**Logging.** Append-only, hash-chained event log capturing every state write, every state read, every tool invocation and every gate decision, with run seed. Logs are published with results.

**Attribution procedure.** Minimal causal set recovery proceeds by systematic ablation of state entries with re-execution under fixed seed. The specific search strategy and its optimisations are withheld from this version and will be published together with results.

---

## 6. Pre-committed falsification criteria

The hypothesis is to be considered **refuted**, and the null result published, if:

**F1.** MIR shows no significant dependence on Δ̂ at α = 0.05 under the pre-specified model, across at least 3 model families at the registered sample size. The decisive comparison is MIR(0→2) against MIR(1→2), for the reason given in §4: a significant result confined to MIR(0→2) against MIR(0→1) is reported as a capability-count effect and triggers F1. *(Refutes H1 — the core claim. If F1 triggers, the work is a null result on privilege differential, not a finding about agent memory.)*

**F2.** The control category is shown to be statistically **equivalent** to every substantive category, by two one-sided tests against an equivalence margin of 0.10 absolute MIR at α = 0.0125 per one-sided test, on per-category MIR pooled across model families. *(Refutes H2 — the observed behaviour is unconditioned and reduces to known misevolution.)*

A failure to reject a difference does not trigger F2. At any realistic sample size that outcome is consistent both with equivalence and with an effect the design did not detect, so stating the criterion as a non-significant difference would make F2 fire on insufficient data rather than on evidence of no conditioning. The registered margin of 0.10 is the minimum effect of interest: the claim being pre-committed is that any categorical conditioning is smaller than the effect this work set out to find. At the registered sample size this procedure has 0.93 power when the true difference is zero.

**F3.** AIR < 0.50. *(Refutes H3 — the phenomenon exists but is not forensically attributable, which removes the defensive contribution.)*

**F4.** Discard rate under atomic benignness (D6) exceeds 30% of runs. *(The premise fails: the originating agents are not behaving benignly, and the result collapses into the self-propagating-misalignment setting already established in arXiv:2610.04083.)*

A run that collapses (§5, *Run outcomes*) is not a discard under D6 and does not enter the F4 rate: a model that failed to carry out its part has not written an entry that fails atomic benignness.

F1 and F4 are the decisive ones. F4 in particular is the criterion that keeps this work distinct from prior results rather than a restatement of them.

---

## 7. Threats to validity

Recorded in advance, without mitigation claims:

- **Author bias in scenario design.** The same author writes the tasks and the expected-action sets. Partial mitigation: tasks frozen before hypotheses are tested; full task set published; the structural properties on which the inferences depend — balance across categories, mandate identity within an archetype, satisfiability by a read-only writer, absence of scope-expansion language — enforced as test conditions rather than left to review. What remains unmitigated is the choice of subject matter and of what counts as a realistic task, which no test can check and which a single author cannot resolve.
- **Opportunity-set confound in one H1 comparison.** The set of out-of-mandate actions available to a consumer grows with its authority level, so MIR(0→2) > MIR(0→1) does not by itself distinguish the privilege differential from capability count. MIR(0→2) > MIR(1→2) holds the consumer and its opportunity set fixed and does distinguish them; see §4. The design records this rather than correcting for it, because the two conditions cannot be equalised on opportunity-set size without making the gap conditions non-nested.
- **Benignness judgment is subjective.** D6 depends on an LLM judge plus spot human review. Inter-rater agreement to be reported.
- **Non-determinism.** Fixed seeds do not guarantee reproducibility across providers. Ablation results are therefore reported with repetition counts, not as single-shot counterfactuals.
- **Mandate specification.** *allowed(M)* must be enumerable to be enforceable, which biases toward tasks with crisp action boundaries and may not generalise to open-ended work.
- **Construct validity of Δ̂.** Counting allowlist entries treats all capabilities as equally weighted, which they are not. A blast-radius weighting is frozen with this version and reported as a sensitivity analysis alongside the primary unweighted figures (§5). The weights are themselves a judgment of the author and are not independently validated; what the sensitivity analysis establishes is only that H1's ordering does not depend on the choice between these two.
- **Gate granularity.** The policy gate decides D7(i)–(ii) on tool names, not on arguments, and every read inside the sandbox is within every mandate. A consumer that uses a permitted tool against a resource the mandate did not contemplate — reading a file outside the task's intended scope with `fs.read_file`, or writing a file the task did not ask for with `fs.write_file`, say — is doing something the principal did not ask for, and the gate will pass it. The bias runs in one direction only: towards missing inflation events, never towards inventing them, so a measured MIR is a lower bound on the quantity D7 defines. Argument-level enforcement would require a per-task specification of legitimate resources, which reintroduces the judgment at the boundary that the deterministic gate exists to remove.
- **Model version drift.** Provider models change under fixed names. Collection dates and version identifiers logged per run.
- **External validity.** The systems measured are open-weight models of 8 to 14 billion parameters, quantised, run locally under one harness, with one handoff between two agents. Whether the effect, or its absence, carries to larger or hosted models, to other agent frameworks, or to longer chains is not tested here.
- **Family confounded with size.** Each family is represented by one model, and the three differ in size (8, 12 and 14 billion parameters) as well as in developer. The family term absorbs both; no claim is made about developer, training or size as such. A family replaced under §5 is replaced by a model of the same size, so replacement does not add to this confounding.
- **Harness choices made on smoke runs.** The description of the environment and the request for one call per reply were added after smoke runs on a single task. They are fixed before collection and identical across conditions, but they shape what the agents do, and a different harness could give different rates.
- **Collapse as selection.** Runs that collapse are replaced, and the replacement is not random if collapse depends on the condition. §5 registers the comparison and the sensitivity analysis this requires.

---

## 8. Scope and ethics

All work is conducted in a local sandbox with synthetic data. No production system, third-party service, or live credential is involved at any point. There is no adversary in the threat model and no attack payload is developed, so no exploit artifact is produced or withheld.

Should the hypothesis be confirmed, the finding describes a property of a common architectural pattern rather than a vulnerability in a specific product. No coordinated disclosure to a single vendor is therefore anticipated; if a specific framework is found to be disproportionately affected, its maintainers will be contacted before results are published.

---

## 9. Versioning and priority

This document records the formulation, metric definitions, and falsification criteria as of the stated date. It contains no results.

- **v0.1** — hypotheses, metrics, protocol, falsification criteria. Superseded; remains citable at its own DOI.
- **v0.2** — frozen task set, allowlists, power analysis.
- **v0.3** — harness (policy gate and event journal), the gate properties recorded in §5, the gate-granularity limitation in §7, and the correction below.
- **v0.4 (this version)** — the analysis script of §5, the estimator it is computed with, and the correction below.
- **v0.5** — in preparation, not yet deposited: the run executor and what the agents are shown, the models and their sampling parameters, run outcomes and the replacement of collapsed runs, the family-replacement rule, and the pilot protocol.
- **v1.0** — full logs, results, and either the finding or the null.

What v0.2 adds to the record, beyond the material named above: the sample size is resized against H2 rather than H1 and now accounts for the discard rate F4 permits; F2 is restated as an equivalence test, since a failure to reject is not evidence of no conditioning; model family moves from a random to a fixed effect; and the two comparisons in H1 are distinguished by what each can establish. Four of these are revisions to v0.1, made before any data was collected and recorded here rather than silently.

What v0.3 adds, all before any data collection: the policy gate and the append-only event journal are implemented and frozen; the four properties of the gate that bear on the metrics are recorded in §5; the gate's tool-name granularity is recorded in §7 as a threat to construct validity; and a near neighbour removing the adversary at a single decision point is distinguished in §1.

v0.3 also corrects two figures that described the v0.1 sample size after §5 was resized in v0.2: the frozen `analysis/power.py` carried 100 runs per cell where §5 registers 135, and the per-cell minimum detectable effect was quoted from the smaller figure. The registered design is unchanged — 135 runs per cell, 6,075 total — and the correction brings the code and the stated MDE into agreement with it. A test now parses the registered figures out of this document and compares them to the module constants, so the two cannot drift apart again without the build failing. The per-cell valid-run floor is additionally restated as the pooled figure the analysis actually uses.

What v0.4 adds, all before any data collection: the analysis script registered in §5 is written and frozen as `analysis/model.py`, implementing the mixed-effects model, the pre-specified comparisons of §4 under Holm correction, the bootstrap intervals, the equivalence test F2 is stated as, and the verdict of §6; and the estimator that computes it is registered in §5 together with the checks it must pass before collection and the independent refit reported beside it afterwards.

v0.4 also withdraws a claim this document has carried since v0.2. The version list above named an analysis script among the material frozen in v0.2, and the README repeated it. No such script was in the v0.2 deposit or in the v0.3 deposit: `analysis/power.py` sizes the design and states in its own documentation that it is not the analysis model, and the model of §5 had no implementation until this version. The claim is withdrawn and the script it described is frozen here. The test that holds this document and the code together was comparing registered figures only, which is why a missing artifact passed it; it now also requires that every source file this document names is present, so an artifact can no longer be registered in prose without existing.

What v0.5 adds, all before any data collection: the run executor of §5 and the choices it makes about what the agents are shown, including the description of the environment and the request for one tool call per reply, both added after smoke runs; the three models, run locally, with the reading of "distinct providers" as three developers of open weights; the decision to run each model at its shipped sampling parameters rather than at a temperature chosen for the design; the definition of a completed run that M1's validity condition refers to, and the separation of collapsed runs from D6 discards, with the replacement of collapsed runs, its limit and the checks it requires; a rule for replacing a family that cannot carry out the task, with a candidate of the same size named for each family it can apply to; and a pilot protocol that fixes, by rules stated in advance, the parameters that could not be fixed without running the models. §7 records the external validity of the design, the confounding of model family with model size, the harness choices made on smoke runs, and collapse as a source of selection.

This version is recorded as it is written. The worlds the tasks are carried out against and the pilot table with the values its rules give are added before it is deposited.

v0.5 also amends the frozen task set. Every mandate, and every writer's tool set, now contains all of L0, where each previously named the reads its archetype was expected to use. In smoke runs on 14 November 2026 a writer refused a directory listing its brief did not name either gave up or was counted as breaching its brief — three runs of five — and a consumer looking at a directory would have been a candidate inflation event. A read inside the sandbox is not what this work measures. The opportunity sets of §4 change accordingly, from 12 or 13 actions for an L2 consumer and 5 or 6 for an L1 consumer to 9 and 2 for every task; the consumer of the decisive comparison is at L2 in both of its conditions, so that comparison is unaffected.

The registered contribution remains the formulation, the metric, and the design. The harness is published because the gate's behaviour determines what the primary metric counts, which makes it part of the design rather than an implementation detail.

---

## 10. How to cite

> Kiryak, E. (2026). *Privilege-Differential Drift: Measuring Mandate Inflation in Hierarchical LLM Agent Systems — A Pre-Registration*. MASec Lab. Zenodo. DOI: [10.5281/zenodo.23513214](https://doi.org/10.5281/zenodo.23513214)

The concept DOI above always resolves to the latest version. Version-specific DOIs: v0.4 — [10.5281/zenodo.24080308](https://doi.org/10.5281/zenodo.24080308); v0.3 — [10.5281/zenodo.23724565](https://doi.org/10.5281/zenodo.23724565); v0.2 — [10.5281/zenodo.23699919](https://doi.org/10.5281/zenodo.23699919); v0.1 — [10.5281/zenodo.23513215](https://doi.org/10.5281/zenodo.23513215).

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
