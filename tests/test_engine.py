"""Tests for the segment-agnostic simulation engine, including a regression check
against a baseline captured from the pre-refactor notebook code."""
from pathlib import Path

import numpy as np
import pytest

from paracellular_transport.config import PHYSICAL_CONSTANTS, TAL_STATE1, TAL_STATE1_5_STRANDS, TAL_STATE2_AVG, TAL_STATE2_AVG_5_STRANDS, TAL_STATE2_ORDER_10B_FIRST, TAL_STATE2_ORDER_1619_FIRST, TAL_STATE2_ORDER_ALTERNATING, TAL_STATE2_PARALLEL, TAL_STATE2_PARALLEL_5_STRANDS, CompartmentSpec, JunctionSpec, Pathway, Scenario, SimulationSettings
from paracellular_transport.engine import build_scenario, find_steady_state, run_scenario, run_simulation, steady_state_for_scenario, steady_state_gap, transepithelial_potential, with_total_time_steps
from paracellular_transport.models import Compartment, Junctions
from paracellular_transport.physics import shared_voltage

BASELINE_PATH = Path(__file__).parent / "data" / "tal_regression_baseline.npz"


class TestBuildScenario:
    def test_wires_compartments_and_junctions_to_matching_specs(self):
        compartments, pathway_junctions = build_scenario(TAL_STATE2_PARALLEL)

        assert set(compartments) == {'A', 'B', 'C', 'D'}
        assert compartments['A'].concentrations['Na'] == TAL_STATE2_PARALLEL.compartments[0].na_conc

        assert set(pathway_junctions) == {'10b', '16_19'}
        first_pathway = TAL_STATE2_PARALLEL.pathways[0]
        junctions = pathway_junctions[first_pathway.name]
        assert len(junctions) == len(first_pathway.junctions)
        for junction, spec in zip(junctions, first_pathway.junctions):
            assert junction.apical is compartments[spec.apical]
            assert junction.basolateral is compartments[spec.basolateral]
            assert junction.permeabilities == {'Na': spec.p_na, 'Cl': spec.p_cl, 'Mg': spec.p_mg}


class TestRunSimulationSinglePathway:
    def test_fixed_compartments_are_exactly_unchanged_while_others_evolve(self):
        scenario = with_total_time_steps(TAL_STATE1, 5)
        result = run_scenario(scenario)

        for ion in ('Na', 'Cl', 'Mg'):
            # A and D are fixed boundary reservoirs: every recorded value must be identical.
            assert result.concentration_history['A'][ion] == [result.concentration_history['A'][ion][0]] * 5
            assert result.concentration_history['D'][ion] == [result.concentration_history['D'][ion][0]] * 5
            # B evolves: it must change by the second recorded step.
            assert result.concentration_history['B'][ion][1] != result.concentration_history['B'][ion][0]

    def test_history_lengths_match_total_time_steps(self):
        scenario = with_total_time_steps(TAL_STATE1, 7)
        result = run_scenario(scenario)
        assert len(result.time_axis) == 7
        assert len(result.concentration_history['B']['Na']) == 7


class TestRunSimulationPathwayCountValidation:
    def _minimal_pathway_junctions(self, n):
        a = Compartment('A', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        b = Compartment('B', na_conc=100.0, cl_conc=100.0, volume=8e-20, mg_conc=0.5)
        return {
            f"pathway_{i}": [Junctions(a, b, p_na=10.0, p_cl=1.0, p_mg=3.0)]
            for i in range(n)
        }

    def test_raises_for_zero_pathways(self):
        with pytest.raises(ValueError):
            run_simulation({}, self._minimal_pathway_junctions(0), SimulationSettings(dt=1e-4, total_time_steps=1, temperature=310))

    def test_raises_for_three_pathways(self):
        with pytest.raises(ValueError):
            run_simulation({}, self._minimal_pathway_junctions(3), SimulationSettings(dt=1e-4, total_time_steps=1, temperature=310))

    def test_raises_when_a_resistance_is_missing_for_two_pathways(self):
        compartments, pathway_junctions = build_scenario(TAL_STATE2_PARALLEL)
        with pytest.raises(ValueError):
            run_simulation(
                compartments, pathway_junctions, TAL_STATE2_PARALLEL.settings,
                pathway_resistances={'10b': 14.0},  # missing '16_19'
            )


class TestSharedVoltageStep:
    """Targets the two-pathway shared-voltage combination step, including the
    reset-to-None-before-recompute detail that was previously a bug (see engine.py)."""

    def _expected_potentials_mV(self, concentrations_at_t, pathways, resistances, constants, temperature):
        """Independently recompute, from a snapshot of concentrations, what the shared
        potential across each junction pair SHOULD be -- without going through
        run_simulation at all, so it can't share its bug."""
        # volume is irrelevant here -- this helper only calls calculate_potentials, which
        # never reads it -- so every compartment gets the same placeholder value.
        comps = {name: Compartment(name, vals['Na'], vals['Cl'], volume=8e-20, mg_conc=vals['Mg']) for name, vals in concentrations_at_t.items()}
        name_1, name_2 = [p.name for p in pathways]
        specs_1, specs_2 = pathways[0].junctions, pathways[1].junctions
        R1, R2 = resistances[name_1], resistances[name_2]

        expected = {name_1: {}, name_2: {}}
        for spec_1, spec_2 in zip(specs_1, specs_2):
            j1 = Junctions(comps[spec_1.apical], comps[spec_1.basolateral], spec_1.p_na, spec_1.p_cl, spec_1.p_mg)
            j2 = Junctions(comps[spec_2.apical], comps[spec_2.basolateral], spec_2.p_na, spec_2.p_cl, spec_2.p_mg)
            U1 = j1.calculate_potentials(constants.R, temperature, constants.F, divalent=True)
            U2 = j2.calculate_potentials(constants.R, temperature, constants.F, divalent=True)
            U_k = shared_voltage(U1, U2, R1, R2)
            key = f"{spec_1.apical}->{spec_1.basolateral}"
            expected[name_1][key] = -U_k * 1000
            expected[name_2][key] = -U_k * 1000
        return expected

    def test_first_timestep_matches_manual_shared_voltage(self):
        scenario = with_total_time_steps(TAL_STATE2_PARALLEL, 1)
        result = run_scenario(scenario)

        initial_concentrations = {
            c.name: {'Na': c.na_conc, 'Cl': c.cl_conc, 'Mg': c.mg_conc} for c in scenario.compartments
        }
        resistances = {p.name: p.resistance for p in scenario.pathways}
        expected = self._expected_potentials_mV(
            initial_concentrations, scenario.pathways, resistances, PHYSICAL_CONSTANTS, scenario.settings.temperature
        )

        for pathway_name, junctions in expected.items():
            for junction_key, expected_value in junctions.items():
                assert result.potential_history[pathway_name][junction_key][0] == pytest.approx(expected_value)

    def test_shared_voltage_tracks_concentrations_at_every_step_not_just_the_first(self):
        """Regression test for the 'stale clamp' bug: if voltage_clamp were not reset
        to None before recomputing each step, the engine would keep replaying step 0's
        shared voltage forever instead of tracking concentrations. Comparing against an
        independent recomputation at EVERY step (not just step 0) catches that."""
        n_steps = 6
        scenario = with_total_time_steps(TAL_STATE2_PARALLEL, n_steps)
        result = run_scenario(scenario)
        resistances = {p.name: p.resistance for p in scenario.pathways}

        for t in range(n_steps):
            concentrations_at_t = {
                comp: {ion: result.concentration_history[comp][ion][t] for ion in ('Na', 'Cl', 'Mg')}
                for comp in result.concentration_history
            }
            expected = self._expected_potentials_mV(
                concentrations_at_t, scenario.pathways, resistances, PHYSICAL_CONSTANTS, scenario.settings.temperature
            )
            for pathway_name, junctions in expected.items():
                for junction_key, expected_value in junctions.items():
                    actual = result.potential_history[pathway_name][junction_key][t]
                    assert actual == pytest.approx(expected_value), f"step {t}, {pathway_name} {junction_key}"


@pytest.fixture(scope="module")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.skip(f"No regression baseline at {BASELINE_PATH}")
    return np.load(BASELINE_PATH)


class TestRegressionAgainstPreRefactorBaseline:
    """Master safety net: the refactor must not change any number the original
    notebook code produced. Baseline captured once from the unmodified pre-refactor
    code at total_time_steps=20 (see the plan / tests/regenerate_baseline.py)."""

    def _check(self, baseline, prefix, scenario, flow_key_by_pathway):
        result = run_scenario(with_total_time_steps(scenario, 20))

        assert np.allclose(result.time_axis, baseline[f"{prefix}_time_axis"])

        for comp, ions in result.concentration_history.items():
            for ion, vals in ions.items():
                assert np.allclose(vals, baseline[f"{prefix}_history_{comp}_{ion}"]), f"{prefix} concentration {comp}/{ion}"

        for pathway_name, junctions in result.flow_history.items():
            base_key = flow_key_by_pathway[pathway_name]
            for junction, ions in junctions.items():
                jkey = junction.replace("->", "_to_")
                for ion, vals in ions.items():
                    assert np.allclose(vals, baseline[f"{prefix}_{base_key}_{jkey}_{ion}"]), f"{prefix} flow {pathway_name}/{junction}/{ion}"

        for pathway_name, junctions in result.potential_history.items():
            base_key = flow_key_by_pathway[pathway_name].replace("flow", "pot")
            for junction, vals in junctions.items():
                jkey = junction.replace("->", "_to_")
                assert np.allclose(vals, baseline[f"{prefix}_{base_key}_{jkey}"]), f"{prefix} potential {pathway_name}/{junction}"

    def test_state1(self, baseline):
        self._check(baseline, "state1", TAL_STATE1, {'10b': 'flow10b'})

    def test_state2_parallel(self, baseline):
        self._check(baseline, "state2_parallel", TAL_STATE2_PARALLEL, {'10b': 'flow10b', '16_19': 'flow1619'})

    def test_state2_avg(self, baseline):
        self._check(baseline, "state2_avg", TAL_STATE2_AVG, {'avg': 'flow10b'})


class TestConcentrationTrajectoryDependsOnVolume:
    """Regression tests for the volume-cancellation fix (see CHANGELOG.md): compartment
    volume previously had zero effect on concentration_history, since engine.py divided
    a flux-derived ion count back out by the exact same shared volume it was multiplied
    by. Each compartment now carries its own volume, and the two legs no longer cancel."""

    def _deltas_after_one_step(self, apical_volume, basolateral_volume):
        apical = Compartment('apical', na_conc=200.0, cl_conc=1.0, volume=apical_volume, mg_conc=1.0)
        basolateral = Compartment('basolateral', na_conc=50.0, cl_conc=1.0, volume=basolateral_volume, mg_conc=1.0)
        junction = Junctions(apical, basolateral, p_na=10.0, p_cl=0.0, p_mg=0.0, voltage_clamp=0.0)
        settings = SimulationSettings(dt=1e-4, total_time_steps=2, temperature=310)
        result = run_simulation(
            {'apical': apical, 'basolateral': basolateral}, {'only': [junction]}, settings,
            ion_list=('Na',), fixed_compartments=(),
        )
        return {
            'apical': result.concentration_history['apical']['Na'][1] - result.concentration_history['apical']['Na'][0],
            'basolateral': result.concentration_history['basolateral']['Na'][1] - result.concentration_history['basolateral']['Na'][0],
        }

    def test_donor_side_change_is_independent_of_its_own_volume(self):
        # apical (200) > basolateral (50), zero clamp -> Na diffuses apical->basolateral,
        # i.e. apical is the donor. A donor's own concentration follows flux*dt regardless
        # of its own volume -- this was already true pre-fix and must stay true.
        default = self._deltas_after_one_step(apical_volume=8e-20, basolateral_volume=8e-20)
        smaller_donor = self._deltas_after_one_step(apical_volume=8e-20 / 10, basolateral_volume=8e-20)
        assert smaller_donor['apical'] == pytest.approx(default['apical'])

    def test_recipient_side_change_scales_with_donor_volume(self):
        # The same transported ion count now converts to a bigger concentration swing
        # on the recipient (basolateral) side when the donor is smaller -- this is the
        # behavior that was previously impossible (volume cancelled out entirely).
        default = self._deltas_after_one_step(apical_volume=8e-20, basolateral_volume=8e-20)
        smaller_donor = self._deltas_after_one_step(apical_volume=8e-20 / 10, basolateral_volume=8e-20)
        assert smaller_donor['basolateral'] == pytest.approx(default['basolateral'] / 10)


class Test5StrandScenariosRunEndToEnd:
    """Confirms the engine is genuinely chain-length-agnostic, not just in theory:
    the 6-compartment/5-junction scenarios must build and run without error."""

    @pytest.mark.parametrize("scenario", [TAL_STATE1_5_STRANDS, TAL_STATE2_PARALLEL_5_STRANDS, TAL_STATE2_AVG_5_STRANDS])
    def test_runs_and_produces_history_for_every_compartment(self, scenario):
        result = run_scenario(with_total_time_steps(scenario, 5))

        assert len(result.time_axis) == 5
        assert set(result.concentration_history) == set(scenario.fixed_compartments) | {'B', 'C', 'D', 'E'}
        for comp, ions in result.concentration_history.items():
            for ion in ('Na', 'Cl', 'Mg'):
                assert len(ions[ion]) == 5

    def test_interior_compartments_evolve_while_boundaries_stay_fixed(self):
        result = run_scenario(with_total_time_steps(TAL_STATE1_5_STRANDS, 5))

        for ion in ('Na', 'Cl', 'Mg'):
            assert result.concentration_history['A'][ion] == [result.concentration_history['A'][ion][0]] * 5
            assert result.concentration_history['F'][ion] == [result.concentration_history['F'][ion][0]] * 5
            assert result.concentration_history['B'][ion][1] != result.concentration_history['B'][ion][0]


class TestStrandOrderScenariosRunEndToEnd:
    """Strand order must actually change the result -- otherwise the three
    scenarios would be an experiment that cannot detect what it is testing."""

    ORDERS = [TAL_STATE2_ORDER_ALTERNATING, TAL_STATE2_ORDER_10B_FIRST, TAL_STATE2_ORDER_1619_FIRST]

    @pytest.mark.parametrize("scenario", ORDERS)
    def test_runs_as_a_single_pathway_chain(self, scenario):
        result = run_scenario(with_total_time_steps(scenario, 5))

        assert list(result.potential_history) == ['chain']
        assert len(result.time_axis) == 5

    def test_same_composition_in_different_orders_gives_different_potentials(self):
        finals = [
            transepithelial_potential(
                run_scenario(with_total_time_steps(scenario, 200)).potential_history['chain']
            )[-1]
            for scenario in self.ORDERS
        ]
        assert len(set(round(v, 6) for v in finals)) == 3

    def test_reversing_the_chain_changes_the_potential(self):
        """The concentration gradient is directional (dilute lumen -> plasma-like
        blood side), so a reversed chain is not an equivalent system."""
        forward, backward = (
            transepithelial_potential(
                run_scenario(with_total_time_steps(scenario, 200)).potential_history['chain']
            )[-1]
            for scenario in (TAL_STATE2_ORDER_10B_FIRST, TAL_STATE2_ORDER_1619_FIRST)
        )
        assert forward != pytest.approx(backward, rel=1e-3)


class TestFindSteadyState:
    """find_steady_state solves the same model run_simulation forward-integrates,
    so these tests don't depend on total_time_steps at all -- the solve is fast
    (root-finding, not a million-step loop) regardless of a scenario's own setting."""

    def test_converges_for_single_pathway(self):
        result = steady_state_for_scenario(TAL_STATE1)
        assert result.success
        assert result.max_abs_rate < 1e-6

    def test_converges_for_two_pathway_shared_voltage(self):
        """Exercises the shared-voltage-combination branch inside _step_deltas."""
        result = steady_state_for_scenario(TAL_STATE2_PARALLEL)
        assert result.success
        assert result.max_abs_rate < 1e-6

    def test_fixed_compartments_are_excluded_from_the_solve(self):
        result = steady_state_for_scenario(TAL_STATE1)
        apical_spec = TAL_STATE1.compartments[0]
        assert result.concentrations['A'] == {'Na': apical_spec.na_conc, 'Cl': apical_spec.cl_conc, 'Mg': apical_spec.mg_conc}

    def test_raises_for_missing_pathway_resistances(self):
        """_prepare_pathways' validation is shared with run_simulation -- same conditions must raise here too."""
        compartments, pathway_junctions = build_scenario(TAL_STATE2_PARALLEL)
        with pytest.raises(ValueError):
            find_steady_state(
                compartments, pathway_junctions, TAL_STATE2_PARALLEL.settings,
                pathway_resistances={'10b': 14.0},  # missing '16_19'
            )

    def test_steady_state_is_a_fixed_point_of_run_simulation(self):
        """Setting every compartment to find_steady_state's solution and then running
        a modest number of run_simulation steps should barely move anything -- proving
        the solved state is a genuine fixed point of the SAME physics run_simulation
        uses (both share _step_deltas), not a separately-reimplemented approximation."""
        compartments, pathway_junctions = build_scenario(TAL_STATE1)
        steady = find_steady_state(
            compartments, pathway_junctions, TAL_STATE1.settings, fixed_compartments=TAL_STATE1.fixed_compartments
        )
        assert steady.success

        # compartments now hold the solved steady state (find_steady_state mutates in place)
        settings = with_total_time_steps(TAL_STATE1, 100).settings
        result = run_simulation(compartments, pathway_junctions, settings, fixed_compartments=TAL_STATE1.fixed_compartments)

        for name in ('B', 'C'):
            for ion in ('Na', 'Cl', 'Mg'):
                first = result.concentration_history[name][ion][0]
                last = result.concentration_history[name][ion][-1]
                assert last == pytest.approx(first, abs=1e-6)


class TestSteadyStateGap:
    def test_gap_shrinks_towards_the_steady_state(self):
        """A short run starting far from steady state (TAL_STATE1's own initial
        condition) should end up closer to the true steady state than it started."""
        steady = steady_state_for_scenario(TAL_STATE1)
        result = run_scenario(with_total_time_steps(TAL_STATE1, 200))

        gap_per_variable, max_gap_over_time = steady_state_gap(result, steady, TAL_STATE1.fixed_compartments)

        assert len(max_gap_over_time) == len(result.time_axis) == 200
        assert max_gap_over_time[-1] < max_gap_over_time[0]
        for key, gaps in gap_per_variable.items():
            assert len(gaps) == 200
            assert gaps[-1] < gaps[0], f"{key} did not get closer to steady state"

    def test_excludes_fixed_compartments(self):
        steady = steady_state_for_scenario(TAL_STATE1)
        result = run_scenario(with_total_time_steps(TAL_STATE1, 5))

        gap_per_variable, _ = steady_state_gap(result, steady, TAL_STATE1.fixed_compartments)

        names_present = {name for name, _ in gap_per_variable}
        assert names_present.isdisjoint(TAL_STATE1.fixed_compartments)

    def test_indexing_at_a_given_total_time_steps_matches_a_run_of_that_length(self):
        """max_gap_over_time[N-1] from a longer run should match running only N steps --
        this is the whole point: checking a candidate total_time_steps without re-running."""
        steady = steady_state_for_scenario(TAL_STATE1)
        long_result = run_scenario(with_total_time_steps(TAL_STATE1, 50))
        short_result = run_scenario(with_total_time_steps(TAL_STATE1, 10))

        _, long_gap = steady_state_gap(long_result, steady, TAL_STATE1.fixed_compartments)
        _, short_gap = steady_state_gap(short_result, steady, TAL_STATE1.fixed_compartments)

        assert long_gap[9] == pytest.approx(short_gap[-1])
