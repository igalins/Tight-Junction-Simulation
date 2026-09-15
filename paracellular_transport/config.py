"""Every hardcoded scientific/numeric value used by this package, in one place.

Contains a generic, segment-agnostic schema (``SimulationSettings``,
``CompartmentSpec``, ``JunctionSpec``, ``Pathway``, ``Scenario``) plus the
concrete data for the Thick Ascending Limb (TAL). A future nephron segment
(e.g. Proximal Tubule) would add its own set of module-level constants and
``Scenario`` instances here, reusing the same schema and the same
``engine.run_simulation`` loop.
"""
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

TAL_SETTINGS = SimulationSettings(dt=0.0001, total_time_steps=1_000_000, temperature=310)

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

# ---- Resistances for kidney tubule, DOI: 10.1016/j.kint.2017.08.029 ----
R_10B = 14.0
R_1619 = 19.0

# ---- Voltage clamp (State 1 only) ----
V_CLAMP_STATE1 = -0.009  # V; represents the 5-13 mV lumen-positive potential, split evenly across the 3 junctions


def _tal_state1() -> Scenario:
    """Early/medullary TAL (mTAL): high luminal concentrations, voltage-clamped, Cldn10b only."""
    compartments = (
        CompartmentSpec('A', na_conc=250, cl_conc=233, volume=TAL_VOLUME, mg_conc=2.0),  # apical, high luminal (medullary side)
        CompartmentSpec('B', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),  # tight-junction compartment
        CompartmentSpec('C', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),  # tight-junction compartment
        CompartmentSpec('D', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),  # basolateral, plasma (fixed, physiological)
    )
    # Voltage clamp of 9 mV on the whole A-D chain, split evenly across the 3 junctions
    # (active transcellular pump contribution).
    clamp = V_CLAMP_STATE1 / 3
    pathway = Pathway(
        name="10b",
        junctions=(
            JunctionSpec('A', 'B', P_NA_10B, P_CL, P_MG_10B, voltage_clamp=clamp),
            JunctionSpec('B', 'C', P_NA_10B, P_CL, P_MG_10B, voltage_clamp=clamp),
            JunctionSpec('C', 'D', P_NA_10B, P_CL, P_MG_10B, voltage_clamp=clamp),
        ),
    )
    return Scenario(name="TAL State 1 (mTAL)", settings=TAL_SETTINGS, compartments=compartments, pathways=(pathway,))


def _tal_state2_parallel() -> Scenario:
    """Late/cortical TAL (cTAL): dilute luminal concentrations, no clamp, Cldn10b + Cldn16/19 in parallel."""
    compartments = (
        CompartmentSpec('A', na_conc=50, cl_conc=45, volume=TAL_VOLUME, mg_conc=0.2),  # apical, dilute (active NaCl reabsorption)
        CompartmentSpec('B', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
        CompartmentSpec('C', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
        CompartmentSpec('D', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
    )
    # Neither pathway is clamped here -- both must be free to compute their own
    # equilibrium potential from their own permeabilities. engine.run_simulation
    # combines them into a shared voltage (resistance-weighted) fresh every timestep.
    pathway_10b = Pathway(
        name="10b",
        junctions=(
            JunctionSpec('A', 'B', P_NA_10B, P_CL, P_MG_10B),
            JunctionSpec('B', 'C', P_NA_10B, P_CL, P_MG_10B),
            JunctionSpec('C', 'D', P_NA_10B, P_CL, P_MG_10B),
        ),
        resistance=R_10B,
    )
    pathway_1619 = Pathway(
        name="16_19",
        junctions=(
            JunctionSpec('A', 'B', P_NA_1619, P_CL, P_MG_1619),
            JunctionSpec('B', 'C', P_NA_1619, P_CL, P_MG_1619),
            JunctionSpec('C', 'D', P_NA_1619, P_CL, P_MG_1619),
        ),
        resistance=R_1619,
    )
    return Scenario(
        name="TAL State 2, parallel claudins (cTAL)",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(pathway_10b, pathway_1619),
    )


def _tal_state2_avg() -> Scenario:
    """Late/cortical TAL (cTAL), single averaged claudin (double permeability) for comparison against parallel mode."""
    compartments = (
        CompartmentSpec('A', na_conc=50, cl_conc=45, volume=TAL_VOLUME, mg_conc=0.2),
        CompartmentSpec('B', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
        CompartmentSpec('C', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
        CompartmentSpec('D', na_conc=145, cl_conc=105, volume=TAL_VOLUME, mg_conc=0.5),
    )
    pathway = Pathway(
        name="avg",
        junctions=(
            JunctionSpec('A', 'B', P_NA_AVG, P_CL_AVG, P_MG_AVG),
            JunctionSpec('B', 'C', P_NA_AVG, P_CL_AVG, P_MG_AVG),
            JunctionSpec('C', 'D', P_NA_AVG, P_CL_AVG, P_MG_AVG),
        ),
    )
    return Scenario(
        name="TAL State 2, averaged claudin (cTAL)",
        settings=TAL_SETTINGS,
        compartments=compartments,
        pathways=(pathway,),
    )


TAL_STATE1 = _tal_state1()
TAL_STATE2_PARALLEL = _tal_state2_parallel()
TAL_STATE2_AVG = _tal_state2_avg()
