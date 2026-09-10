"""Regenerate tests/data/tal_regression_baseline.npz from the current package.

Run this and commit the resulting .npz whenever you INTENTIONALLY change a
hardcoded scientific value in paracellular_transport/config.py (a permeability,
resistance, concentration, etc.) -- the regression test in test_engine.py
compares against this file, so it needs refreshing to reflect the new,
deliberately-changed numbers. Do not run this to "fix" a failing regression
test without first confirming the change in output was intentional.

Usage:
    python tests/regenerate_baseline.py
"""
import dataclasses
from pathlib import Path

import numpy as np

from paracellular_transport.config import TAL_STATE1, TAL_STATE2_AVG, TAL_STATE2_PARALLEL
from paracellular_transport.engine import run_scenario

TOTAL_TIME_STEPS = 20  # keep in sync with tests/test_engine.py's with_total_time_steps(..., 20) calls
OUTPUT_PATH = Path(__file__).parent / "data" / "tal_regression_baseline.npz"


def with_total_time_steps(scenario, total_time_steps):
    return dataclasses.replace(scenario, settings=dataclasses.replace(scenario.settings, total_time_steps=total_time_steps))


def flatten(prefix, out, result, flow_key_by_pathway):
    out[f"{prefix}_time_axis"] = np.array(result.time_axis)
    for comp, ions in result.concentration_history.items():
        for ion, vals in ions.items():
            out[f"{prefix}_history_{comp}_{ion}"] = np.array(vals)
    for pathway_name, junctions in result.flow_history.items():
        base_key = flow_key_by_pathway[pathway_name]
        for junction, ions in junctions.items():
            jkey = junction.replace("->", "_to_")
            for ion, vals in ions.items():
                out[f"{prefix}_{base_key}_{jkey}_{ion}"] = np.array(vals)
    for pathway_name, junctions in result.potential_history.items():
        base_key = flow_key_by_pathway[pathway_name].replace("flow", "pot")
        for junction, vals in junctions.items():
            jkey = junction.replace("->", "_to_")
            out[f"{prefix}_{base_key}_{jkey}"] = np.array(vals)


def main():
    out = {}
    flatten("state1", out, run_scenario(with_total_time_steps(TAL_STATE1, TOTAL_TIME_STEPS)), {'10b': 'flow10b'})
    flatten(
        "state2_parallel", out, run_scenario(with_total_time_steps(TAL_STATE2_PARALLEL, TOTAL_TIME_STEPS)),
        {'10b': 'flow10b', '16_19': 'flow1619'},
    )
    flatten("state2_avg", out, run_scenario(with_total_time_steps(TAL_STATE2_AVG, TOTAL_TIME_STEPS)), {'avg': 'flow10b'})

    np.savez(OUTPUT_PATH, **out)
    print(f"Saved {len(out)} arrays to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
