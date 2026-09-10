"""Tests for the segment-agnostic simulation engine, including a regression check
against a baseline captured from the pre-refactor notebook code."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from paracellular_transport.config import PHYSICAL_CONSTANTS, TAL_STATE1, TAL_STATE2_AVG, TAL_STATE2_PARALLEL, CompartmentSpec, JunctionSpec, Pathway, Scenario, SimulationSettings
from paracellular_transport.engine import build_scenario, run_scenario, run_simulation
from paracellular_transport.models import Compartment, Junctions
from paracellular_transport.physics import shared_voltage

BASELINE_PATH = Path(__file__).parent / "data" / "tal_regression_baseline.npz"


def with_total_time_steps(scenario: Scenario, total_time_steps: int) -> Scenario:
    return dataclasses.replace(scenario, settings=dataclasses.replace(scenario.settings, total_time_steps=total_time_steps))


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
        a = Compartment('A', na_conc=100.0, cl_conc=100.0, mg_conc=0.5)
        b = Compartment('B', na_conc=100.0, cl_conc=100.0, mg_conc=0.5)
        return {
            f"pathway_{i}": [Junctions(a, b, p_na=10.0, p_cl=1.0, p_mg=3.0)]
            for i in range(n)
        }

    def test_raises_for_zero_pathways(self):
        with pytest.raises(ValueError):
            run_simulation({}, self._minimal_pathway_junctions(0), SimulationSettings(dt=1e-4, total_time_steps=1, volume=8e-20, temperature=310))

    def test_raises_for_three_pathways(self):
        with pytest.raises(ValueError):
            run_simulation({}, self._minimal_pathway_junctions(3), SimulationSettings(dt=1e-4, total_time_steps=1, volume=8e-20, temperature=310))

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
        comps = {name: Compartment(name, vals['Na'], vals['Cl'], vals['Mg']) for name, vals in concentrations_at_t.items()}
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
