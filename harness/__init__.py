# SPDX-License-Identifier: Apache-2.0

"""Experimental harness for the registered design.

`journal` implements the append-only, hash-chained event log of §5.
`gate` implements the deterministic policy gate of D1 and the D7(i)-(ii)
detection of §3.
`executor` runs a writer and a consumer through both against a simulated
world. It is not part of any deposited version yet.

The D7(iii) ablation procedure and the D6 benignness review are not in this
package yet.
"""
