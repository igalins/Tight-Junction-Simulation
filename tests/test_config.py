"""Guards against derived config constants silently drifting out of sync if someone edits one but not the others."""
import pytest

from paracellular_transport.config import (
    CLAUDIN_PROFILES,
    P_CL,
    P_CL_AVG,
    P_MG_10B,
    P_MG_1619,
    P_MG_AVG,
    P_NA_10B,
    P_NA_1619,
    P_NA_AVG,
    TAL_STATE1,
    TAL_STATE1_5_STRANDS,
    TAL_STATE2_AVG_5_STRANDS,
    TAL_STATE2_ORDER_10B_FIRST,
    TAL_STATE2_ORDER_1619_FIRST,
    TAL_STATE2_ORDER_ALTERNATING,
    TAL_STATE2_PARALLEL_5_STRANDS,
    V_CLAMP_STATE1,
)


def test_averaged_permeabilities_stay_derived_from_the_two_claudin_types():
    assert P_NA_AVG == pytest.approx(P_NA_10B + P_NA_1619)
    assert P_MG_AVG == pytest.approx(P_MG_10B + P_MG_1619)
    assert P_CL_AVG == pytest.approx(2 * P_CL)


def test_state1_voltage_clamps_sum_to_the_total_clamp():
    clamps = [j.voltage_clamp for pathway in TAL_STATE1.pathways for j in pathway.junctions]
    assert sum(clamps) == pytest.approx(V_CLAMP_STATE1)


class Test5StrandScenarios:
    @pytest.mark.parametrize(
        "scenario,n_pathways",
        [
            (TAL_STATE1_5_STRANDS, 1),
            (TAL_STATE2_PARALLEL_5_STRANDS, 2),
            (TAL_STATE2_AVG_5_STRANDS, 1),
        ],
    )
    def test_has_six_compartments_and_five_junctions_per_pathway(self, scenario, n_pathways):
        assert len(scenario.compartments) == 6
        assert len(scenario.pathways) == n_pathways
        for pathway in scenario.pathways:
            assert len(pathway.junctions) == 5

    @pytest.mark.parametrize("scenario", [TAL_STATE1_5_STRANDS, TAL_STATE2_PARALLEL_5_STRANDS, TAL_STATE2_AVG_5_STRANDS])
    def test_fixed_compartments_are_first_and_last(self, scenario):
        assert scenario.fixed_compartments == ('A', 'F')

    def test_state1_voltage_clamps_still_sum_to_the_total_clamp(self):
        clamps = [j.voltage_clamp for pathway in TAL_STATE1_5_STRANDS.pathways for j in pathway.junctions]
        assert sum(clamps) == pytest.approx(V_CLAMP_STATE1)


class TestStrandOrderScenarios:
    """The three strand-order scenarios must differ only in arrangement -- if they
    ever differ in composition too, the comparison stops isolating order."""

    ORDERS = [TAL_STATE2_ORDER_ALTERNATING, TAL_STATE2_ORDER_10B_FIRST, TAL_STATE2_ORDER_1619_FIRST]

    @pytest.mark.parametrize("scenario", ORDERS)
    def test_is_a_single_five_strand_pathway(self, scenario):
        assert len(scenario.pathways) == 1
        assert len(scenario.pathways[0].junctions) == 5
        assert len(scenario.compartments) == 6
        assert scenario.fixed_compartments == ('A', 'F')

    def test_all_three_orders_share_the_same_composition(self):
        compositions = [
            sorted((j.p_na, j.p_cl, j.p_mg) for j in scenario.pathways[0].junctions)
            for scenario in self.ORDERS
        ]
        assert compositions[0] == compositions[1] == compositions[2]

    def test_composition_is_three_10b_and_two_1619(self):
        for scenario in self.ORDERS:
            profiles = [(j.p_na, j.p_cl, j.p_mg) for j in scenario.pathways[0].junctions]
            assert profiles.count(CLAUDIN_PROFILES['10b']) == 3
            assert profiles.count(CLAUDIN_PROFILES['16_19']) == 2

    def test_the_two_blocked_orders_are_exact_reversals(self):
        forward = [(j.p_na, j.p_cl, j.p_mg) for j in TAL_STATE2_ORDER_10B_FIRST.pathways[0].junctions]
        backward = [(j.p_na, j.p_cl, j.p_mg) for j in TAL_STATE2_ORDER_1619_FIRST.pathways[0].junctions]
        assert forward == backward[::-1]

    @pytest.mark.parametrize("scenario", ORDERS)
    def test_junctions_form_a_connected_chain(self, scenario):
        junctions = scenario.pathways[0].junctions
        assert [j.apical for j in junctions] == ['A', 'B', 'C', 'D', 'E']
        assert [j.basolateral for j in junctions] == ['B', 'C', 'D', 'E', 'F']
