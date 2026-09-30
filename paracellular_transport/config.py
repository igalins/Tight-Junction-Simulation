"""Every hardcoded scientific/numeric value used by this package, in one place.

Contains a generic, segment-agnostic schema (``SimulationSettings``,
``CompartmentSpec``, ``JunctionSpec``, ``Pathway``, ``Scenario``) plus the
concrete data for the Thick Ascending Limb (TAL). A future nephron segment
(e.g. Proximal Tubule) would add its own set of module-level constants and
``Scenario`` instances here, reusing the same schema and the same
``engine.run_simulation`` loop.
"""
import string
from dataclasses import dataclass
from scipy.constants import Avogadro, gas_constant, physical_constants


@dataclass(frozen=True)
class PhysicalConstants:
    """Universal physical constants used throughout the model."""
    R: float = gas_constant
    F: float = physical_constants["Faraday constant"][0]
    Avogadro: float = Avogadro


PHYSICAL_CONSTANTS = PhysicalConstants()


@dataclass(frozen=True)
class SimulationSettings:
    """Numeric parameters governing how a simulation is run.

    Attributes:
        dt: Timestep duration (s).
        total_time_steps: Number of timesteps to simulate.
        temperature: Absolute temperature (K).
    """
    dt: float
    total_time_steps: int
    temperature: float


@dataclass(frozen=True)
class CompartmentSpec:
    """Initial state for one compartment in a scenario's chain.

    Attributes:
        name: Compartment label (e.g. 'A', 'B').
        na_conc: Initial Na+ concentration.
        cl_conc: Initial Cl- concentration.
        volume: Compartment volume (liters).
        mg_conc: Initial Mg2+ concentration.
    """
    name: str
    na_conc: float
    cl_conc: float
    volume: float
    mg_conc: float = 0.0


@dataclass(frozen=True)
class JunctionSpec:
    """One junction between two named compartments, within a pathway.

    Attributes:
        apical: Name of the apical-side compartment.
        basolateral: Name of the basolateral-side compartment.
        p_na: Na+ permeability.
        p_cl: Cl- permeability.
        p_mg: Mg2+ permeability.
        voltage_clamp: Optional fixed potential (volts) for this junction.
    """
    apical: str
    basolateral: str
    p_na: float
    p_cl: float
    p_mg: float
    voltage_clamp: float | None = None


@dataclass(frozen=True)
class Pathway:
    """One parallel wiring of a compartment chain with a single permeability profile.

    E.g. in a "parallel claudin" scenario, the same A-B-C-D chain is wired up
    twice — once entirely with Cldn10b permeabilities, once entirely with
    Cldn16/19 permeabilities — and each wiring is one ``Pathway``. A
    single-claudin or averaged-claudin scenario has just one ``Pathway``.

    Attributes:
        name: Pathway label (e.g. '10b', '16_19', 'avg').
        junctions: The junctions making up this pathway's chain.
        resistance: Required only when two pathways are combined via
            ``physics.shared_voltage`` (see ``engine.run_simulation``).
    """
    name: str
    junctions: tuple[JunctionSpec, ...]
    resistance: float | None = None


@dataclass(frozen=True)
class Scenario:
    """A complete, ready-to-run simulation setup.

    Attributes:
        name: Human-readable scenario label.
        settings: Numeric simulation parameters.
        compartments: Initial compartment states, in chain order.
        pathways: One or two ``Pathway``s sharing the same compartments.
        fixed_compartments: Names of compartments held constant (not updated)
            across timesteps — the boundary reservoirs.
        ion_list: Ions to simulate.
        divalent: Whether to use the Mg2+-extended potential equation.
    """
    name: str
    settings: SimulationSettings
    compartments: tuple[CompartmentSpec, ...]
    pathways: tuple[Pathway, ...]
    fixed_compartments: tuple[str, ...] = ('A', 'D')
    ion_list: tuple[str, ...] = ('Na', 'Cl', 'Mg')
    divalent: bool = True


# ============================================================================
# Thick Ascending Limb (TAL) data
# ============================================================================

TAL_SETTINGS = SimulationSettings(dt=0.0001, total_time_steps=5_000_000, temperature=310)

# Currently uniform across all TAL compartments -- compartments could be given
# different volumes independently, since each CompartmentSpec carries its own.
TAL_VOLUME = 8e-20

# ---- Permeabilities ----
# Cldn10b: PNa:PCl = 10:1; PMg:PCl = 3:1
# Cldn16/19: PNa:PCl = 2.5:1; PMg:PCl = 8.5:1
P_CL = 1.0  # baseline

P_NA_10B = 10.0
P_MG_10B = 3.0

P_NA_1619 = 2.5
P_MG_1619 = 8.5

# Not divided by 2: the averaged-claudin pathway is compared against
# Cldn10b + Cldn16/19 combined, i.e. 2x a single averaged claudin.
P_NA_AVG = P_NA_10B + P_NA_1619
P_MG_AVG = P_MG_10B + P_MG_1619
P_CL_AVG = P_CL * 2

# Per-strand permeability profiles (p_na, p_cl, p_mg), for chains whose strands
# are not all the same claudin -- see ``_tal_state2_strand_order``.
CLAUDIN_PROFILES = {
    '10b': (P_NA_10B, P_CL, P_MG_10B),
    '16_19': (P_NA_1619, P_CL, P_MG_1619),
}

# ---- Resistances for kidney tubule, DOI: 10.1016/j.kint.2017.08.029 ----
R_10B = 14.0
R_1619 = 19.0

# ---- Voltage clamp (State 1 only) ----
V_CLAMP_STATE1 = -0.009  # V; represents the 5-13 mV lumen-positive potential, split evenly across the chain's junctions


# ---- Generic compartment-chain builders ----
# Each JunctionSpec in a pathway's chain represents one physical tight-junction
# (claudin) strand in series. A chain of n_strands strands has n_strands + 1
# compartments: the apical/luminal reservoir, n_strands - 1 interior
# "tight-junction compartment" pockets between strands, and the basolateral/
# plasma reservoir -- both reservoirs are the scenario's fixed_compartments.

def _chain_compartment_names(n_strands):
    """Compartment names for a chain of n_strands strands, e.g. ('A','B','C','D') for 3."""
    return tuple(string.ascii_uppercase[:n_strands + 1])


def _chain_compartments(n_strands, apical_na, apical_cl, apical_mg):
    """The apical reservoir at the given concentrations, followed by every remaining
    compartment (tj compartments + the basolateral reservoir) at the shared
    plasma-like concentrations used throughout the TAL scenarios."""
    names = _chain_compartment_names(n_strands)
    apical = CompartmentSpec(names[0], na_conc=apical_na, cl_conc=apical_cl, volume=TAL_VOLUME, mg_conc=apical_mg)
    rest = tuple(
        CompartmentSpec(name, na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5)
        for name in names[1:]
    )
    return (apical,) + rest


def _chain_junctions(n_strands, p_na, p_cl, p_mg, voltage_clamp=None):
    """One JunctionSpec per consecutive compartment pair, all sharing the same permeabilities."""
    names = _chain_compartment_names(n_strands)
    return tuple(
        JunctionSpec(names[i], names[i + 1], p_na, p_cl, p_mg, voltage_clamp=voltage_clamp)
        for i in range(n_strands)
    )


def _tal_state1(n_strands=3) -> Scenario:
    """Early/medullary TAL (mTAL): high luminal concentrations, voltage-clamped, Cldn10b only."""
    compartments = _chain_compartments(n_strands, apical_na=250, apical_cl=233, apical_mg=2.0)
    names = _chain_compartment_names(n_strands)
    # Voltage clamp of 9 mV on the whole chain, split evenly across its junctions
    # (active transcellular pump contribution).
    clamp = V_CLAMP_STATE1 / n_strands
    pathway = Pathway(name="10b", junctions=_chain_junctions(n_strands, P_NA_10B, P_CL, P_MG_10B, voltage_clamp=clamp))
    return Scenario(
        name=f"TAL State 1 (mTAL), {n_strands} strands",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(pathway,),
        fixed_compartments=(names[0], names[-1]),
    )


def _tal_state2_parallel(n_strands=3) -> Scenario:
    """Late/cortical TAL (cTAL): dilute luminal concentrations, no clamp, Cldn10b + Cldn16/19 in parallel."""
    compartments = _chain_compartments(n_strands, apical_na=50, apical_cl=45, apical_mg=0.2)
    names = _chain_compartment_names(n_strands)
    # Neither pathway is clamped here -- both must be free to compute their own
    # equilibrium potential from their own permeabilities. engine.run_simulation
    # combines them into a shared voltage (resistance-weighted) fresh every timestep.
    pathway_10b = Pathway(name="10b", junctions=_chain_junctions(n_strands, P_NA_10B, P_CL, P_MG_10B), resistance=R_10B)
    pathway_1619 = Pathway(name="16_19", junctions=_chain_junctions(n_strands, P_NA_1619, P_CL, P_MG_1619), resistance=R_1619)
    return Scenario(
        name=f"TAL State 2, parallel claudins (cTAL), {n_strands} strands",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(pathway_10b, pathway_1619),
        fixed_compartments=(names[0], names[-1]),
    )


def _tal_state2_avg(n_strands=3) -> Scenario:
    """Late/cortical TAL (cTAL), single averaged claudin (double permeability) for comparison against parallel mode."""
    compartments = _chain_compartments(n_strands, apical_na=50, apical_cl=45, apical_mg=0.2)
    names = _chain_compartment_names(n_strands)
    pathway = Pathway(name="avg", junctions=_chain_junctions(n_strands, P_NA_AVG, P_CL_AVG, P_MG_AVG))
    return Scenario(
        name=f"TAL State 2, averaged claudin (cTAL), {n_strands} strands",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(pathway,),
        fixed_compartments=(names[0], names[-1]),
    )


def _tal_state2_strand_order(strand_order) -> Scenario:
    """Late/cortical TAL (cTAL) chain whose strands are individually Cldn10b or
    Cldn16/19, named apical-to-basolateral (lumen first).

    One pathway, not two: every strand position carries a single claudin, so
    there is no second parallel route to combine into a shared voltage. Compare
    orders of the *same composition* against each other to isolate the effect
    of arrangement alone.

    Args:
        strand_order: Claudin key per strand, apical to basolateral, e.g.
            ``('10b', '16_19', '10b')``. Keys index ``CLAUDIN_PROFILES``.

    Returns:
        The corresponding ``Scenario``.
    """
    n_strands = len(strand_order)
    compartments = _chain_compartments(n_strands, apical_na=50, apical_cl=45, apical_mg=0.2)
    names = _chain_compartment_names(n_strands)
    junctions = tuple(
        JunctionSpec(names[i], names[i + 1], *CLAUDIN_PROFILES[claudin])
        for i, claudin in enumerate(strand_order)
    )
    return Scenario(
        name=f"TAL State 2, strand order {' -> '.join(strand_order)}",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(Pathway(name='chain', junctions=junctions),),
        fixed_compartments=(names[0], names[-1]),
    )


TAL_STATE1 = _tal_state1()
TAL_STATE2_PARALLEL = _tal_state2_parallel()
TAL_STATE2_AVG = _tal_state2_avg()

TAL_STATE1_5_STRANDS = _tal_state1(n_strands=5)
TAL_STATE2_PARALLEL_5_STRANDS = _tal_state2_parallel(n_strands=5)
TAL_STATE2_AVG_5_STRANDS = _tal_state2_avg(n_strands=5)

# Same composition (3x Cldn10b, 2x Cldn16/19) in three different arrangements,
# so any difference between them is down to order alone. The first-named strand
# is the apical/luminal one; the last two are exact reversals of each other.
TAL_STATE2_ORDER_ALTERNATING = _tal_state2_strand_order(('10b', '16_19', '10b', '16_19', '10b'))
TAL_STATE2_ORDER_10B_FIRST = _tal_state2_strand_order(('10b', '10b', '10b', '16_19', '16_19'))
TAL_STATE2_ORDER_1619_FIRST = _tal_state2_strand_order(('16_19', '16_19', '10b', '10b', '10b'))
