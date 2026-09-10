"""Segment-agnostic simulation engine.

Builds live ``Compartment``/``Junctions`` objects from a ``config.Scenario``,
runs the timestep loop, and records history. Nothing here is specific to any
nephron segment — a future segment (e.g. Proximal Tubule) is added purely by
defining a new ``Scenario`` in ``config.py``; this loop does not change.
"""
from dataclasses import dataclass, field

from .config import PHYSICAL_CONSTANTS, PhysicalConstants, Scenario, SimulationSettings
from .models import Compartment, Junctions
from .physics import shared_voltage


@dataclass
class SimulationResult:
    """Recorded output of a ``run_simulation`` call.

    Attributes:
        time_axis: Simulation time at each recorded step (s).
        concentration_history: compartment name -> ion -> list of concentrations over time.
        flow_history: pathway name -> junction key ("A->B") -> ion -> list of ion-count changes over time.
        potential_history: pathway name -> junction key -> list of membrane potentials (mV) over time.
    """
    time_axis: list = field(default_factory=list)
    concentration_history: dict = field(default_factory=dict)
    flow_history: dict = field(default_factory=dict)
    potential_history: dict = field(default_factory=dict)


def build_scenario(scenario: Scenario):
    """Instantiate live ``Compartment``/``Junctions`` objects from a ``Scenario``'s frozen specs.

    Args:
        scenario: The scenario to build.

    Returns:
        A tuple ``(compartments, pathway_junctions)``:
            compartments: dict of compartment name -> ``Compartment``.
            pathway_junctions: dict of pathway name -> list of ``Junctions``, in chain order.
    """
    compartments = {c.name: Compartment(c.name, c.na_conc, c.cl_conc, c.mg_conc) for c in scenario.compartments}

    pathway_junctions = {}
    for pathway in scenario.pathways:
        junctions = [
            Junctions(
                compartments[j.apical],
                compartments[j.basolateral],
                j.p_na, j.p_cl, j.p_mg,
                voltage_clamp=j.voltage_clamp,
            )
            for j in pathway.junctions
        ]
        pathway_junctions[pathway.name] = junctions

    return compartments, pathway_junctions


def run_simulation(
    compartments,
    pathway_junctions,
    settings: SimulationSettings,
    constants: PhysicalConstants = PHYSICAL_CONSTANTS,
    ion_list=('Na', 'Cl', 'Mg'),
    divalent=True,
    fixed_compartments=('A', 'D'),
    pathway_resistances=None,
) -> SimulationResult:
    """Run the main paracellular-transport simulation loop.

    Supports 1 or 2 parallel pathways sharing the same compartments. With 2
    pathways, their potentials are recombined into one shared voltage every
    timestep via ``physics.shared_voltage`` and both pathways' junctions are
    clamped to it, reflecting the physical constraint that two parallel
    routes between the same two compartments must share one boundary
    voltage. With 1 pathway, no combination step happens (this also covers
    a single voltage-clamped pathway, since its junctions simply keep
    returning their fixed clamp value every step).

    Args:
        compartments: dict of compartment name -> ``Compartment``, e.g. from ``build_scenario``.
        pathway_junctions: dict of pathway name -> list of ``Junctions`` (1 or 2 entries).
        settings: Numeric simulation parameters (dt, total_time_steps, volume, temperature).
        constants: Physical constants (R, F, Avogadro).
        ion_list: Ions to simulate.
        divalent: Whether to use the Mg2+-extended potential equation.
        fixed_compartments: Names of compartments excluded from concentration
            updates (boundary reservoirs, e.g. the luminal and blood sides).
        pathway_resistances: dict of pathway name -> resistance; required
            when ``pathway_junctions`` has exactly 2 entries.

    Returns:
        A ``SimulationResult`` with the recorded histories.

    Raises:
        ValueError: If ``pathway_junctions`` has neither 1 nor 2 entries, or
            (with 2 entries) a resistance is missing for either pathway.
            ``shared_voltage`` is a pairwise combination and does not
            generalize past 2 pathways.
    """
    pathway_names = list(pathway_junctions.keys())
    if len(pathway_names) not in (1, 2):
        raise ValueError(
            f"run_simulation supports 1 or 2 pathways (shared_voltage is a pairwise "
            f"combination), got {len(pathway_names)}: {pathway_names}"
        )

    two_pathway = len(pathway_names) == 2
    if two_pathway:
        if pathway_resistances is None or any(name not in pathway_resistances for name in pathway_names):
            raise ValueError(
                f"pathway_resistances must include a resistance for every pathway "
                f"when combining 2 pathways, got pathways {pathway_names} and "
                f"pathway_resistances={pathway_resistances}"
            )
        name_1, name_2 = pathway_names
        junctions_1, junctions_2 = pathway_junctions[name_1], pathway_junctions[name_2]
        R1, R2 = pathway_resistances[name_1], pathway_resistances[name_2]

    R, F, T = constants.R, constants.F, settings.temperature

    flow_history = {
        name: {f"{j.apical.name}->{j.basolateral.name}": {ion: [] for ion in ion_list} for j in junctions}
        for name, junctions in pathway_junctions.items()
    }
    potential_history = {
        name: {f"{j.apical.name}->{j.basolateral.name}": [] for j in junctions}
        for name, junctions in pathway_junctions.items()
    }
    concentration_history = {name: {ion: [] for ion in ion_list} for name in compartments}
    time_axis = []

    for t in range(settings.total_time_steps):
        time_axis.append(t * settings.dt)

        # save compartment concentrations
        for name, comp in compartments.items():
            for ion in ion_list:
                concentration_history[name][ion].append(comp.concentrations[ion])

        deltas = {name: {ion: 0.0 for ion in ion_list} for name in compartments}

        # Combine both pathways' independently-computed potentials into one shared
        # voltage, then clamp both to it. Resetting voltage_clamp to None first is
        # required: calculate_potentials() early-returns self.voltage_clamp when set,
        # so without the reset this would just replay the PREVIOUS timestep's shared
        # voltage instead of recomputing from this timestep's live concentrations.
        if two_pathway:
            for j1, j2 in zip(junctions_1, junctions_2):
                j1.voltage_clamp = None
                j2.voltage_clamp = None
                U1 = j1.calculate_potentials(R, T, F, divalent=divalent)
                U2 = j2.calculate_potentials(R, T, F, divalent=divalent)

                U_k = shared_voltage(U1, U2, R1, R2)
                j1.voltage_clamp = U_k
                j2.voltage_clamp = U_k

        # compute fluxes for every pathway and accumulate deltas
        for name, junctions in pathway_junctions.items():
            for j in junctions:
                membrane_pot = j.calculate_potentials(R, T, F, divalent=divalent)
                potential_history[name][f"{j.apical.name}->{j.basolateral.name}"].append(-membrane_pot * 1000)
                changes = j.calculate_fluxes(R, T, F, ion_list, membrane_pot, settings.dt, settings.volume)
                for ion, amount in changes.items():
                    conc_change = amount / (constants.Avogadro * settings.volume)
                    deltas[j.apical.name][ion] -= conc_change
                    deltas[j.basolateral.name][ion] += conc_change
                    flow_history[name][f"{j.apical.name}->{j.basolateral.name}"][ion].append(amount)

        # update every compartment except the fixed boundary reservoirs
        for name, comp in compartments.items():
            if name not in fixed_compartments:
                for ion in ion_list:
                    comp.concentrations[ion] += deltas[name][ion]

    return SimulationResult(
        time_axis=time_axis,
        concentration_history=concentration_history,
        flow_history=flow_history,
        potential_history=potential_history,
    )


def run_scenario(scenario: Scenario) -> SimulationResult:
    """Build and run a ``Scenario`` in one call — the main entry point for notebooks/tests.

    Args:
        scenario: The scenario to run.

    Returns:
        The resulting ``SimulationResult``.
    """
    compartments, pathway_junctions = build_scenario(scenario)
    pathway_resistances = (
        {p.name: p.resistance for p in scenario.pathways} if len(scenario.pathways) == 2 else None
    )
    return run_simulation(
        compartments,
        pathway_junctions,
        scenario.settings,
        ion_list=scenario.ion_list,
        divalent=scenario.divalent,
        fixed_compartments=scenario.fixed_compartments,
        pathway_resistances=pathway_resistances,
    )


def transepithelial_potential(potential_history_for_one_pathway):
    """Sum one pathway's per-junction potentials into a transepithelial value at each timestep.

    Args:
        potential_history_for_one_pathway: dict of junction key -> list of
            potentials, e.g. ``result.potential_history['10b']``.

    Returns:
        A list summing all junctions' potentials at each timestep. Works for
        any chain length/order, not just a specific A->B->C->D layout.
    """
    return [sum(values) for values in zip(*potential_history_for_one_pathway.values())]
