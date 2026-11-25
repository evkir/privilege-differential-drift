# SPDX-License-Identifier: Apache-2.0

"""Experimental harness for the registered design.

`journal` implements the append-only, hash-chained event log of §5.
`gate` implements the deterministic policy gate of D1 and the D7(i)-(ii)
detection of §3.
`executor` runs a writer and a consumer through both against a simulated
world. It is registered in v0.5.

`ablation` carries out the D7(iii) ablation: the consumer re-run on every
subset of its state, and the minimal causal sets. The D6 benignness review
is not in this package yet.

`collect` runs collection: the cells of the design in the order §5 registers,
the replacement of collapsed runs, and the ablation of each closed block. It is
registered in v0.6.
"""
