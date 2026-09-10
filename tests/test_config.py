"""Guards against derived config constants silently drifting out of sync if someone edits one but not the others."""
import pytest

from paracellular_transport.config import (
    P_CL,
    P_CL_AVG,
    P_MG_10B,
    P_MG_1619,
    P_MG_AVG,
    P_NA_10B,
    P_NA_1619,
    P_NA_AVG,
    TAL_STATE1,
    V_CLAMP_STATE1,
)


def test_averaged_permeabilities_stay_derived_from_the_two_claudin_types():
    assert P_NA_AVG == pytest.approx(P_NA_10B + P_NA_1619)
    assert P_MG_AVG == pytest.approx(P_MG_10B + P_MG_1619)
    assert P_CL_AVG == pytest.approx(2 * P_CL)


def test_state1_voltage_clamps_sum_to_the_total_clamp():
    clamps = [j.voltage_clamp for pathway in TAL_STATE1.pathways for j in pathway.junctions]
    assert sum(clamps) == pytest.approx(V_CLAMP_STATE1)
