# SPDX-License-Identifier: Apache-2.0

"""The registered figures in PREREGISTRATION.md must equal the module constants.

Written by Claude (Opus 5, Anthropic) at the author's request.

Every other test of `analysis/power.py` validates the power formulas against
Monte Carlo simulation of the test statistic. None of them can catch a constant
that disagrees with the registered prose, because they all import that constant
and assert against it — `assert h2.n_per_arm == N_PER_CELL` is true whatever
`N_PER_CELL` happens to be.

That gap is not hypothetical. §5 was resized in v0.2 and the frozen module was
left carrying the v0.1 figure, so the published table described a sample size
the document no longer registered, and 124 passing tests reported nothing. This
module closes the class rather than the instance: it parses the registered
figures out of the document and compares them to what the code actually uses, so
the two cannot drift apart again without the build failing.

The document is the authority. When one of these fails, the question to ask is
what §5 registers, not what the module would prefer.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "analysis"))

from power import (  # noqa: E402
    BASELINE_MIR,
    DISCARD_CEILING,
    MDE_REGISTERED,
    N_CATEGORIES,
    N_MODEL_FAMILIES,
    N_PER_CELL,
    REGISTERED_DESIGN,
)

DOC = pathlib.Path(__file__).resolve().parents[1] / "PREREGISTRATION.md"
N_GAP_CONDITIONS = 3


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def _one(pattern: str, what: str) -> str:
    found = re.findall(pattern, _doc())
    assert len(found) == 1, f"{what}: matched {len(found)} times in §5, expected exactly 1"
    return found[0]


def test_collection_target_matches() -> None:
    target = int(_one(r"\*\*(\d+) runs per cell\*\*", "collection target"))
    assert N_PER_CELL == target, (
        f"§5 registers {target} runs per cell; analysis/power.py uses "
        f"N_PER_CELL = {N_PER_CELL}"
    )


def test_total_runs_match() -> None:
    total = int(_one(r"\*\*([\d,]+) runs total\*\*", "total runs").replace(",", ""))
    implied = N_PER_CELL * N_CATEGORIES * N_MODEL_FAMILIES * N_GAP_CONDITIONS
    assert implied == total, f"§5 states {total} runs total; the constants imply {implied}"


def test_baseline_and_effect_of_interest_match() -> None:
    assert BASELINE_MIR == float(_one(r"baseline MIR of (0\.\d+)", "baseline MIR"))
    assert MDE_REGISTERED == float(
        _one(r"minimum effect of interest of (0\.\d+) absolute", "minimum effect of interest")
    )


def test_discard_ceiling_matches_f4() -> None:
    ceiling = int(_one(r"exceeds (\d+)% of runs", "F4 discard ceiling"))
    assert DISCARD_CEILING == ceiling / 100.0


def test_pooled_valid_runs_survive_the_worst_discard_rate() -> None:
    """§5's sizing argument rests on the pooled figure, not the per-cell one.

    H2 is tested on per-category MIR pooled across model families (§4), so the
    quantity that has to survive attrition is the pooled arm.
    """
    required = int(
        _one(r"requires \*\*(\d+) valid runs per category arm\*\*", "valid-run requirement")
    )
    delivered = int(_one(r"leaves \*\*(\d+) valid runs per category arm\*\*", "valid runs delivered"))
    assert required == delivered, (
        f"§5 requires {required} valid runs per arm and claims to leave {delivered}"
    )
    actual = int(N_PER_CELL * N_MODEL_FAMILIES * (1.0 - DISCARD_CEILING))
    assert actual >= required, (
        f"{N_PER_CELL} runs per cell across {N_MODEL_FAMILIES} families leaves {actual} "
        f"valid runs per arm at the {DISCARD_CEILING:.0%} discard rate; §5 requires {required}"
    )


def test_per_cell_mde_quoted_in_section_5_matches_the_module() -> None:
    """The quoted per-cell MDE is a computed figure and goes stale silently."""
    quoted = float(
        _one(r"smallest detectable effect is (0\.\d+) absolute", "per-cell minimum detectable effect")
    )
    cell = next(c for c in REGISTERED_DESIGN if c.label.startswith("H2 per category, single cell"))
    assert round(cell.mde, 3) == quoted, (
        f"§5 quotes a per-cell MDE of {quoted}; the module computes {cell.mde:.3f}"
    )


def test_no_row_claims_to_be_the_registered_h2_analysis_at_one_cell() -> None:
    """H2 is registered on the pooled arm; a per-cell row must not claim otherwise.

    The label matters because the table is printed for readers: a row reading
    "as registered" beside the single-cell power would tell them the registered
    design is underpowered when it is the pooled arm that is registered.
    """
    for cell in REGISTERED_DESIGN:
        if cell.hypothesis == "H2" and cell.n_per_arm == N_PER_CELL:
            assert "as registered" not in cell.label, (
                f"row {cell.label!r} reports a single cell but is labelled as the registered analysis"
            )
