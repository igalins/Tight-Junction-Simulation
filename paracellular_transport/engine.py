"""Segment-agnostic simulation engine.

Builds live ``Compartment``/``Junctions`` objects from a ``config.Scenario``,
runs the timestep loop, and records history. Nothing here is specific to any
nephron segment — a future segment (e.g. Proximal Tubule) is added purely by
defining a new ``Scenario`` in ``config.py``; this loop does not change.
"""
import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

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


@dataclass
class SteadyStateResult:
    """Result of a ``find_steady_state`` call.

    Attributes:
        concentrations: compartment name -> ion -> concentration at the solved
            state (includes fixed compartments, held at their initial values).
        success: Whether the underlying root-finder reports convergence.
        message: The root-finder's own status message.
        rates: compartment name -> ion -> d(concentration)/dt at the solved
            state, for every non-fixed compartment -- how fast each
            concentration is still drifting there. Exactly 0 at a true fixed
            point; see ``find_steady_state``'s docstring for how to read
            this when it isn't exactly 0.
        max_abs_rate: max(abs(rate)) across every non-fixed compartment/ion --
            a single at-a-glance convergence summary, in the same
            concentration-per-second units as the compartments themselves.
    """
    concentrations: dict
    success: bool
    message: str
    rates: dict
    max_abs_rate: float


@dataclass
class MixingCurve:
    """Transepithelial potential as a function of the permeability mixing fraction
    ``alpha``, plus the specific points the averaged and parallel scenarios occupy.

    See ``permeability_mixing_curve`` for how these are produced and what
    ``alpha`` means. All potentials are transepithelial (summed over the chain)
    and in mV, in the same sign convention ``run_simulation`` records.

    Attributes:
        alphas: The mixing fractions sampled, from 0 to 1.
        potentials: The Goldman-family potential at each alpha -- the curve.
        chord: Straight-line interpolation between the two endpoint potentials,
            i.e. what mixing WOULD give if the potential were linear in alpha.
        label_a: Name of the pathway at alpha = 1.
        label_b: Name of the pathway at alpha = 0.
        u_a: Potential of pathway A alone (alpha = 1).
        u_b: Potential of pathway B alone (alpha = 0).
        alpha_shared: The chord position the parallel scenario's
            resistance-weighted shared voltage corresponds to,
            ``R_b / (R_a + R_b)``.
        u_shared: The parallel scenario's shared voltage -- ``chord`` at ``alpha_shared``.
        u_averaged: The averaged-claudin scenario's potential -- the *curve* at alpha = 0.5.
        u_chord_half: The chord at alpha = 0.5, i.e. the naive "simple average
            of the two potentials" reference both other points are measured against.
        curvature_gap: ``u_averaged - u_chord_half`` -- how far the curve bulges
            off its own chord. This is the nonlinearity of the Goldman-family
            equation itself.
        weighting_shift: ``u_shared - u_chord_half`` -- how far the resistance
            weighting moves the parallel scenario along the chord, away from a
            simple 50/50 average.
        net_gap: ``u_averaged - u_shared`` -- the observed averaged-vs-parallel
            difference, equal to ``curvature_gap - weighting_shift``.
    """
    alphas: list
    potentials: list
    chord: list
    label_a: str
    label_b: str
    u_a: float
    u_b: float
    alpha_shared: float
    u_shared: float
    u_averaged: float
    u_chord_half: float
    curvature_gap: float
    weighting_shift: float
    net_gap: float


def permeability_mixing_curve(
    scenario: Scenario,
    concentrations=None,
    n_points=201,
    constants: PhysicalConstants = PHYSICAL_CONSTANTS,
) -> MixingCurve:
    """Trace the transepithelial potential as the two pathways' permeabilities are
    blended, to show why the averaged-claudin scenario and the parallel scenario
    disagree.

    Defines a mixing fraction ``alpha`` that interpolates the permeabilities of
    the scenario's two pathways -- ``P(alpha) = alpha * P_a + (1 - alpha) * P_b``,
    so alpha = 0 is pathway B alone and alpha = 1 is pathway A alone -- and
    evaluates the model's own potential equation along it. Both competing
    scenarios then sit on this single axis:

    * The **averaged-claudin** scenario is the *curve* at alpha = 0.5. Its config
      sums the two pathways' permeabilities rather than averaging them, but the
      Goldman-family potential is scale-invariant in permeability (scaling every
      P by the same factor leaves the potential unchanged), so summed and mean
      permeabilities give an identical potential.
    * The **parallel** scenario is the *chord* at ``alpha_shared``, because
      ``physics.shared_voltage`` is a linear resistance-weighted blend of the two
      pathways' separately-computed potentials.

    The averaged-vs-parallel difference therefore splits into two independent
    causes, both returned: ``curvature_gap`` (the curve is not its own chord)
    and ``weighting_shift`` (the resistances do not weight the two pathways
    50/50).

    Evaluates both scenarios at one shared set of concentrations, which isolates
    the permeability nonlinearity from the separate, much smaller fact that the
    two scenarios settle at slightly different steady states.

    Args:
        scenario: A two-pathway scenario with resistances on both pathways,
            e.g. ``config.TAL_STATE2_PARALLEL``.
        concentrations: compartment name -> ion -> concentration to evaluate at.
            Defaults to the scenario's initial concentrations; pass a
            ``SteadyStateResult.concentrations`` to evaluate at steady state
            instead.
        n_points: Number of alpha samples across [0, 1].
        constants: Physical constants (R, F, Avogadro).

    Returns:
        A ``MixingCurve``.

    Raises:
        ValueError: If ``scenario`` does not have exactly 2 pathways, or either
            pathway is missing a resistance.
    """
    if len(scenario.pathways) != 2:
        raise ValueError(
            f"permeability_mixing_curve compares exactly 2 pathways, got "
            f"{len(scenario.pathways)}: {[p.name for p in scenario.pathways]}"
        )
    pathway_a, pathway_b = scenario.pathways
    if pathway_a.resistance is None or pathway_b.resistance is None:
        raise ValueError(
            f"Both pathways need a resistance to locate the parallel scenario on the "
            f"chord, got {pathway_a.name}={pathway_a.resistance}, {pathway_b.name}={pathway_b.resistance}"
        )

    R, F, T = constants.R, constants.F, scenario.settings.temperature

    if concentrations is None:
        concentrations = {
            c.name: {'Na': c.na_conc, 'Cl': c.cl_conc, 'Mg': c.mg_conc}
            for c in scenario.compartments
        }
    volumes = {c.name: c.volume for c in scenario.compartments}

    compartments = {
        name: Compartment(name, conc['Na'], conc['Cl'], volume=volumes[name], mg_conc=conc['Mg'])
        for name, conc in concentrations.items()
    }
    # One live junction per position in the chain; its permeabilities are
    # overwritten per alpha, so the potential comes from the same
    # Junctions.calculate_potentials the simulation loop uses.
    chain = [
        (Junctions(compartments[spec_a.apical], compartments[spec_a.basolateral], 0.0, 0.0, 0.0), spec_a, spec_b)
        for spec_a, spec_b in zip(pathway_a.junctions, pathway_b.junctions)
    ]

    def transepithelial_at(alpha):
        total = 0.0
        for junction, spec_a, spec_b in chain:
            junction.permeabilities = {
                'Na': alpha * spec_a.p_na + (1 - alpha) * spec_b.p_na,
                'Cl': alpha * spec_a.p_cl + (1 - alpha) * spec_b.p_cl,
                'Mg': alpha * spec_a.p_mg + (1 - alpha) * spec_b.p_mg,
            }
            total += -junction.calculate_potentials(R, T, F, divalent=scenario.divalent) * 1000
        return total

    alphas = np.linspace(0.0, 1.0, n_points).tolist()
    potentials = [transepithelial_at(alpha) for alpha in alphas]

    u_b, u_a = transepithelial_at(0.0), transepithelial_at(1.0)
    chord = [u_b + alpha * (u_a - u_b) for alpha in alphas]

    u_shared = shared_voltage(u_a, u_b, pathway_a.resistance, pathway_b.resistance)
    u_averaged = transepithelial_at(0.5)
    u_chord_half = 0.5 * (u_a + u_b)

    return MixingCurve(
        alphas=alphas,
        potentials=potentials,
        chord=chord,
        label_a=pathway_a.name,
        label_b=pathway_b.name,
        u_a=u_a,
        u_b=u_b,
        alpha_shared=pathway_b.resistance / (pathway_a.resistance + pathway_b.resistance),
        u_shared=u_shared,
        u_averaged=u_averaged,
        u_chord_half=u_chord_half,
        curvature_gap=u_averaged - u_chord_half,
        weighting_shift=u_shared - u_chord_half,
        net_gap=u_averaged - u_shared,
    )


def build_scenario(scenario: Scenario):
    """Instantiate live ``Compartment``/``Junctions`` objects from a ``Scenario``'s frozen specs.

    Args:
        scenario: The scenario to build.

    Returns:
        A tuple ``(compartments, pathway_junctions)``:
            compartments: dict of compartment name -> ``Compartment``.
            pathway_junctions: dict of pathway name -> list of ``Junctions``, in chain order.
    """
    compartments = {
        c.name: Compartment(c.name, c.na_conc, c.cl_conc, volume=c.volume, mg_conc=c.mg_conc)
        for c in scenario.compartments
    }

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


def _prepare_pathways(pathway_junctions, pathway_resistances):
    """Validate the pathway count/resistances once, and pre-extract everything the
    per-instant shared-voltage combination needs.

    Shared by ``run_simulation``'s loop (called once before it starts) and
    ``find_steady_state`` (called once before its repeated root-finding
    evaluations), so the count/resistance validation and the per-pathway
    lookups it does aren't redone on every single step/evaluation.

    Args:
        pathway_junctions: dict of pathway name -> list of ``Junctions`` (1 or 2 entries).
        pathway_resistances: dict of pathway name -> resistance; required
            when ``pathway_junctions`` has exactly 2 entries.

    Returns:
        A dict with key ``'two_pathway'`` (bool); when ``True``, also has
        ``'junctions_1'``, ``'junctions_2'``, ``'R1'``, ``'R2'``.

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

    if len(pathway_names) == 1:
        return {'two_pathway': False}

    if pathway_resistances is None or any(name not in pathway_resistances for name in pathway_names):
        raise ValueError(
            f"pathway_resistances must include a resistance for every pathway "
            f"when combining 2 pathways, got pathways {pathway_names} and "
            f"pathway_resistances={pathway_resistances}"
        )
    name_1, name_2 = pathway_names
    return {
        'two_pathway': True,
        'junctions_1': pathway_junctions[name_1],
        'junctions_2': pathway_junctions[name_2],
        'R1': pathway_resistances[name_1],
        'R2': pathway_resistances[name_2],
    }


def _step_deltas(compartments, pathway_junctions, prepared, R, T, F, ion_list, divalent, dt, avogadro):
    """Compute this instant's per-compartment/ion concentration deltas (the change
    over one timestep of duration ``dt``), plus each junction's membrane potential
    and transported ion amount -- the physics shared by ``run_simulation``'s
    per-timestep loop and ``find_steady_state``'s root-finding (which divides
    these deltas by ``dt`` to get a rate; ``dt`` itself doesn't affect where the
    steady state is, only how big a single simulated timestep would move towards it).

    Args:
        compartments: dict of compartment name -> ``Compartment``.
        pathway_junctions: dict of pathway name -> list of ``Junctions``.
        prepared: This pathway setup, from ``_prepare_pathways``.
        R: Gas constant.
        T: Absolute temperature (K).
        F: Faraday constant.
        ion_list: Ion keys to process, e.g. ('Na', 'Cl', 'Mg').
        divalent: Whether to use the Mg2+-extended potential equation.
        dt: Timestep duration (s).
        avogadro: Avogadro constant.

    Returns:
        A tuple ``(deltas, potentials, flows)``:
            deltas: compartment name -> ion -> concentration change for this dt.
            potentials: pathway name -> junction key -> membrane potential (mV).
            flows: pathway name -> junction key -> ion -> ion-count change for this instant.
    """
    deltas = {name: {ion: 0.0 for ion in ion_list} for name in compartments}
    potentials = {name: {} for name in pathway_junctions}
    flows = {name: {} for name in pathway_junctions}

    # Combine both pathways' independently-computed potentials into one shared
    # voltage, then clamp both to it. Resetting voltage_clamp to None first is
    # required: calculate_potentials() early-returns self.voltage_clamp when set,
    # so without the reset this would just replay the PREVIOUS instant's shared
    # voltage instead of recomputing from the current live concentrations.
    if prepared['two_pathway']:
        for j1, j2 in zip(prepared['junctions_1'], prepared['junctions_2']):
            j1.voltage_clamp = None
            j2.voltage_clamp = None
            U1 = j1.calculate_potentials(R, T, F, divalent=divalent)
            U2 = j2.calculate_potentials(R, T, F, divalent=divalent)

            U_k = shared_voltage(U1, U2, prepared['R1'], prepared['R2'])
            j1.voltage_clamp = U_k
            j2.voltage_clamp = U_k

    # compute fluxes for every pathway and accumulate deltas
    for name, junctions in pathway_junctions.items():
        for j in junctions:
            key = f"{j.apical.name}->{j.basolateral.name}"
            membrane_pot = j.calculate_potentials(R, T, F, divalent=divalent)
            potentials[name][key] = -membrane_pot * 1000
            changes = j.calculate_fluxes(R, T, F, ion_list, membrane_pot, dt)
            flows[name][key] = changes
            for ion, amount in changes.items():
                deltas[j.apical.name][ion] -= amount / (avogadro * j.apical.volume)
                deltas[j.basolateral.name][ion] += amount / (avogadro * j.basolateral.volume)

    return deltas, potentials, flows


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
        settings: Numeric simulation parameters (dt, total_time_steps, temperature).
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
    prepared = _prepare_pathways(pathway_junctions, pathway_resistances)
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

        deltas, potentials, flows = _step_deltas(
            compartments, pathway_junctions, prepared, R, T, F, ion_list, divalent, settings.dt, constants.Avogadro
        )

        for name, junction_potentials in potentials.items():
            for key, val in junction_potentials.items():
                potential_history[name][key].append(val)
        for name, junction_flows in flows.items():
            for key, ion_amounts in junction_flows.items():
                for ion, amount in ion_amounts.items():
                    flow_history[name][key][ion].append(amount)

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


def with_total_time_steps(scenario: Scenario, total_time_steps: int) -> Scenario:
    """A copy of ``scenario`` with a different ``total_time_steps``.

    Useful for checking how far a run of a given length is from steady state
    (see ``steady_state_gap``) without hand-writing the nested
    ``dataclasses.replace`` needed to reach ``settings.total_time_steps``
    through two frozen dataclasses.

    Args:
        scenario: The scenario to copy.
        total_time_steps: The new step count.

    Returns:
        A new ``Scenario``, identical to ``scenario`` except for
        ``settings.total_time_steps``.
    """
    return dataclasses.replace(scenario, settings=dataclasses.replace(scenario.settings, total_time_steps=total_time_steps))


def find_steady_state(
    compartments,
    pathway_junctions,
    settings: SimulationSettings,
    constants: PhysicalConstants = PHYSICAL_CONSTANTS,
    ion_list=('Na', 'Cl', 'Mg'),
    divalent=True,
    fixed_compartments=('A', 'D'),
    pathway_resistances=None,
    method="hybr",
) -> SteadyStateResult:
    """Solve directly for the fixed point of the model (d(concentration)/dt = 0 for
    every non-fixed compartment/ion), instead of forward-integrating
    ``run_simulation`` for a large but arbitrary ``total_time_steps`` and hoping
    it has gotten close enough.

    Reuses the exact same per-instant physics as ``run_simulation``'s loop (via
    ``_prepare_pathways``/``_step_deltas``), so the fixed point found here is
    guaranteed to be a fixed point of the same model the timestep loop
    simulates, not a separately-reimplemented approximation of it.

    Solves in log-concentration space (the root-finder's actual free variables
    are ``log(concentration)``, exponentiated back before every physics
    evaluation) rather than concentration space directly: ``scipy.optimize.root``
    is unconstrained and would otherwise be free to try negative trial
    concentrations while searching, which ``nernst_equation`` rejects outright
    (``ValueError`` for any non-positive concentration). Working in log-space
    makes every trial value positive automatically, with no explicit bounds needed.

    Args:
        compartments: dict of compartment name -> ``Compartment``, e.g. from
            ``build_scenario``. Mutated in place: every compartment ends up
            holding its solved concentrations (or, if the solver fails to
            converge, its last-tried ones).
        pathway_junctions: dict of pathway name -> list of ``Junctions`` (1 or 2 entries).
        settings: Numeric simulation parameters -- only ``temperature`` affects
            the physics here; ``dt``/``total_time_steps`` don't affect where
            the steady state is (see ``rates`` on ``SteadyStateResult``).
        constants: Physical constants (R, F, Avogadro).
        ion_list: Ions to solve for.
        divalent: Whether to use the Mg2+-extended potential equation.
        fixed_compartments: Names of compartments excluded from the solve
            (boundary reservoirs, held fixed at their current concentrations).
        pathway_resistances: dict of pathway name -> resistance; required
            when ``pathway_junctions`` has exactly 2 entries.
        method: ``scipy.optimize.root`` method.

    Returns:
        A ``SteadyStateResult``.

    Raises:
        ValueError: Same conditions as ``run_simulation`` (pathway count/resistances).
    """
    prepared = _prepare_pathways(pathway_junctions, pathway_resistances)
    R, F, T = constants.R, constants.F, settings.temperature

    free_vars = [
        (name, ion)
        for name in compartments
        if name not in fixed_compartments
        for ion in ion_list
    ]
    y0 = np.log([compartments[name].concentrations[ion] for name, ion in free_vars])

    def residuals(y):
        for (name, ion), log_conc in zip(free_vars, y):
            compartments[name].concentrations[ion] = np.exp(log_conc)
        deltas, _, _ = _step_deltas(
            compartments, pathway_junctions, prepared, R, T, F, ion_list, divalent, settings.dt, constants.Avogadro
        )
        return [deltas[name][ion] / settings.dt for name, ion in free_vars]

    sol = root(residuals, y0, method=method)

    # Leave every compartment holding the solved (or, on failure, last-tried) concentrations.
    for (name, ion), log_conc in zip(free_vars, sol.x):
        compartments[name].concentrations[ion] = np.exp(log_conc)

    rates = {}
    for (name, ion), residual in zip(free_vars, sol.fun):
        rates.setdefault(name, {})[ion] = float(residual)

    return SteadyStateResult(
        concentrations={name: dict(comp.concentrations) for name, comp in compartments.items()},
        success=bool(sol.success),
        message=sol.message,
        rates=rates,
        max_abs_rate=float(np.max(np.abs(sol.fun))) if len(sol.fun) else 0.0,
    )


def steady_state_for_scenario(scenario: Scenario, method="hybr") -> SteadyStateResult:
    """Build and solve a ``Scenario`` for its steady state in one call.

    Args:
        scenario: The scenario to solve.
        method: ``scipy.optimize.root`` method.

    Returns:
        The resulting ``SteadyStateResult``.
    """
    compartments, pathway_junctions = build_scenario(scenario)
    pathway_resistances = (
        {p.name: p.resistance for p in scenario.pathways} if len(scenario.pathways) == 2 else None
    )
    return find_steady_state(
        compartments,
        pathway_junctions,
        scenario.settings,
        ion_list=scenario.ion_list,
        divalent=scenario.divalent,
        fixed_compartments=scenario.fixed_compartments,
        pathway_resistances=pathway_resistances,
        method=method,
    )


def steady_state_gap(result: SimulationResult, steady: SteadyStateResult, fixed_compartments):
    """How far a forward ``run_simulation``/``run_scenario`` result's recorded
    trajectory is from a ``find_steady_state``/``steady_state_for_scenario``
    solution, at every recorded timestep.

    Since ``result`` already records every compartment's concentration at
    every timestep, this answers "was ``total_time_steps`` enough?" for any
    step count up to ``result``'s own length by simply indexing into the
    returned lists (e.g. ``max_gap_over_time[total_time_steps - 1]``) --
    without needing to re-run the simulation for each candidate step count.
    To check a step count *beyond* what ``result`` already covers, run a
    longer simulation first (e.g. via ``with_total_time_steps``).

    Args:
        result: A ``SimulationResult``, e.g. from ``run_scenario``.
        steady: The corresponding ``SteadyStateResult`` for the same
            scenario (``total_time_steps`` doesn't affect where the true
            steady state is, so this can come from a ``find_steady_state``
            call with any ``settings``).
        fixed_compartments: Names of compartments to exclude -- the boundary
            reservoirs, held constant, have no meaningful "distance to
            steady state" since they never change.

    Returns:
        A tuple ``(gap_per_variable, max_gap_over_time)``:
            gap_per_variable: (compartment name, ion) -> list of relative
                distances (``abs(value - steady_value) / steady_value``) at
                each recorded timestep.
            max_gap_over_time: list of the largest gap across every
                compartment/ion, at each recorded timestep -- the single
                worst-converged quantity at each point in time.
    """
    free_vars = [
        (name, ion)
        for name, ions in result.concentration_history.items()
        if name not in fixed_compartments
        for ion in ions
    ]

    gap_per_variable = {}
    for name, ion in free_vars:
        steady_value = steady.concentrations[name][ion]
        gap_per_variable[(name, ion)] = [
            abs(value - steady_value) / steady_value for value in result.concentration_history[name][ion]
        ]

    max_gap_over_time = [max(gap_per_variable[key][t] for key in free_vars) for t in range(len(result.time_axis))]

    return gap_per_variable, max_gap_over_time


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
