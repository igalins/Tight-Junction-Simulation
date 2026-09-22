"""Plotting helpers for simulation results.

All functions here take history/time_axis data (as produced by
``engine.run_simulation``) rather than a specific scenario, so they work
unchanged for any nephron segment. Every function that can save a figure
takes an optional ``save_path`` — nothing is written to disk unless the
caller explicitly asks for it.
"""
import matplotlib.pyplot as plt

ION_LABELS = {'Na': 'Na$^+$', 'Cl': 'Cl$^-$', 'Mg': 'Mg$^{2+}$'}


def plot_ion_dynamics(history, time_axis, save_path=None):
    """Plot Na+, Cl-, and Mg2+ concentrations for all compartments over time.

    Args:
        history: dict of compartment name -> ion -> list of concentrations,
            e.g. ``result.concentration_history``.
        time_axis: List of simulation times.
        save_path: If given, save the figure to this path.
    """
    fig, (ax_na, ax_cl, ax_mg) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for comp_name, ions in history.items():
        ax_na.plot(time_axis, ions['Na'], label=f'Na+ {comp_name}')
    ax_na.set_ylabel('Na+ (mol/L)')
    ax_na.set_title('Sodium Concentration Profile')
    ax_na.legend(loc='center right', fontsize='small')
    ax_na.grid(True, alpha=0.3)

    for comp_name, ions in history.items():
        ax_cl.plot(time_axis, ions['Cl'], label=f'Cl- {comp_name}')
    ax_cl.set_ylabel('Cl- (mol/L)')
    ax_cl.set_title('Chloride Concentration Profile')
    ax_cl.legend(loc='center right', fontsize='small')
    ax_cl.grid(True, alpha=0.3)

    for comp_name, ions in history.items():
        ax_mg.plot(time_axis, ions['Mg'], label=f'Mg2+ {comp_name}')
    ax_mg.set_ylabel('Mg2+ (mol/L)')
    ax_mg.set_xlabel('Time (a.u.)')
    ax_mg.set_title('Magnesium Concentration Profile')
    ax_mg.legend(loc='center right', fontsize='small')
    ax_mg.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_flow_dynamics(flow_history, time_axis, title=None, save_path=None):
    """Plot the net movement (flux * dt) of ions between junctions, for one pathway.

    Args:
        flow_history: dict of junction key -> ion -> list of ion-count
            changes, e.g. ``result.flow_history['10b']``.
        time_axis: List of simulation times.
        title: Optional figure super-title.
        save_path: If given, save the figure to this path.
    """
    fig, (ax_na, ax_cl, ax_mg) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    for j_name, ions in flow_history.items():
        ax_na.plot(time_axis, ions['Na'], label=f'Flow {j_name}')
    ax_na.set_ylabel('Na+ Net Flow (arbitrary unit)')
    ax_na.set_title('Sodium Flow Rate Between Compartments')
    ax_na.axhline(0, color='black', lw=1, ls='--')
    ax_na.legend(loc='upper right', fontsize='small')
    ax_na.grid(True, alpha=0.3)

    for j_name, ions in flow_history.items():
        ax_cl.plot(time_axis, ions['Cl'], label=f'Flow {j_name}')
    ax_cl.set_ylabel('Cl- Net Flow (arbitrary unit)')
    ax_cl.set_title('Chloride Flow Rate Between Compartments')
    ax_cl.axhline(0, color='black', lw=1, ls='--')
    ax_cl.legend(loc='upper right', fontsize='small')
    ax_cl.grid(True, alpha=0.3)

    for j_name, ions in flow_history.items():
        if len(ions['Mg']) == 0:
            continue
        ax_mg.plot(time_axis, ions['Mg'], label=f'Flow {j_name}')
    ax_mg.set_ylabel('Mg2+ Net Flow (arbitrary unit)')
    ax_mg.set_xlabel('Time (a.u.)')
    ax_mg.set_title('Magnesium Flow Rate Between Compartments')
    ax_mg.axhline(0, color='black', lw=1, ls='--')
    ax_mg.legend(loc='upper right', fontsize='small')
    ax_mg.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_membrane_potential(transepithelial_potential, time_axis, title=None, save_path=None):
    """Plot a summed transepithelial potential over time.

    Args:
        transepithelial_potential: List of potentials, e.g. from
            ``engine.transepithelial_potential``.
        time_axis: List of simulation times.
        title: Optional figure super-title.
        save_path: If given, save the figure to this path.
    """
    plt.figure(figsize=(12, 6))
    if title:
        plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.plot(time_axis, transepithelial_potential)
    plt.ylabel('Transepithelial Potential (mV)')
    plt.xlabel('Time (a.u.)')
    plt.title('Total Transepithelial Potential (A→D)')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_steady_state_convergence(time_axis, max_gap_over_time, title=None, save_path=None):
    """Plot the largest relative distance to steady state over time, on a log scale.

    Args:
        time_axis: List of simulation times, e.g. ``result.time_axis``.
        max_gap_over_time: List of the largest per-timestep relative gap
            across every free compartment/ion, from ``engine.steady_state_gap``.
        title: Optional plot title.
        save_path: If given, save the figure to this path.
    """
    plt.figure(figsize=(10, 4))
    plt.semilogy(time_axis, max_gap_over_time)
    plt.ylabel('Max relative distance to steady state')
    plt.xlabel('Time (a.u.)')
    plt.title(title or 'Convergence to Steady State')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_permeability_mixing_curve(mixing_curve, title=None, save_path=None):
    """Plot a ``engine.MixingCurve``: the potential against the permeability mixing
    fraction, with the averaged and parallel scenarios marked and the distance
    between them annotated.

    The solid curve is the model's actual potential; the dashed line is the
    straight chord between the two pure-claudin endpoints. The averaged scenario
    sits on the curve at alpha = 0.5, the parallel scenario on the chord at
    ``alpha_shared`` -- so the vertical distance between the two markers is the
    averaged-vs-parallel discrepancy, and the plot shows it splitting into the
    curve's bulge off its chord plus the resistance weighting's sideways shift.

    Args:
        mixing_curve: A ``engine.MixingCurve``, from ``engine.permeability_mixing_curve``.
        title: Optional plot title.
        save_path: If given, save the figure to this path.
    """
    c = mixing_curve

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.plot(c.alphas, c.potentials, lw=2.2, color='C0', label='Actual potential of a mixed junction (nonlinear)')
    ax.plot(c.alphas, c.chord, lw=1.6, ls='--', color='gray', label='Linear blend of the two pure potentials (chord)')

    ax.plot(0.0, c.u_b, 'o', ms=7, color='C7')
    ax.plot(1.0, c.u_a, 'o', ms=7, color='C7')
    ax.annotate(f'{c.label_b} only\n{c.u_b:.2f} mV', xy=(0.0, c.u_b), xytext=(8, -28),
                textcoords='offset points', fontsize=9, color='dimgray')
    ax.annotate(f'{c.label_a} only\n{c.u_a:.2f} mV', xy=(1.0, c.u_a), xytext=(-20, 14),
                textcoords='offset points', fontsize=9, color='dimgray', ha='right')

    ax.plot(0.5, c.u_chord_half, 'x', ms=9, mew=2, color='gray',
            label=f'Naive 50/50 average of the two potentials ({c.u_chord_half:.2f} mV)')
    ax.plot(c.alpha_shared, c.u_shared, 's', ms=10, color='C2', zorder=5,
            label=f'Parallel scenario: chord at $\\alpha$={c.alpha_shared:.3f} ({c.u_shared:.2f} mV)')
    ax.plot(0.5, c.u_averaged, 'o', ms=10, color='C3', zorder=5,
            label=f'Averaged scenario: curve at $\\alpha$=0.5 ({c.u_averaged:.2f} mV)')

    ax.axhline(c.u_averaged, color='C3', ls=':', lw=1, alpha=0.6)
    ax.axhline(c.u_shared, color='C2', ls=':', lw=1, alpha=0.6)
    ax.axhline(c.u_chord_half, color='gray', ls=':', lw=1, alpha=0.5)

    # The averaged-vs-parallel distance, drawn where the two curves leave room.
    x_gap = 0.82
    ax.annotate('', xy=(x_gap, c.u_averaged), xytext=(x_gap, c.u_shared),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.8))
    ax.text(x_gap + 0.015, 0.5 * (c.u_averaged + c.u_shared), f'{c.net_gap:+.2f} mV',
            fontsize=11, fontweight='bold', va='center')

    decomposition = (
        f'curvature (nonlinearity)   {c.curvature_gap:+.2f} mV\n'
        f'resistance weighting       {c.weighting_shift:+.2f} mV\n'
        f'net (averaged - parallel)  {c.net_gap:+.2f} mV'
    )
    ax.text(0.015, 0.975, decomposition, transform=ax.transAxes, va='top', ha='left',
            fontsize=9.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray'))

    ax.set_xlabel(f'$\\alpha$   (fraction of {c.label_a} in the permeability mix)')
    ax.set_ylabel('Transepithelial potential (mV)')
    ax.set_title(title or 'Why the averaged scenario reports a higher potential than the parallel one')
    ax.set_xlim(-0.03, 1.03)
    # Headroom so the endpoint annotations sit inside the axes.
    span = max(c.potentials) - min(c.potentials)
    ax.set_ylim(min(c.potentials) - 0.16 * span, max(c.potentials) + 0.12 * span)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def total_flow(flow_history, ion):
    """Sum one pathway's flux for one ion across all its junctions, at each timestep.

    Args:
        flow_history: dict of junction key -> ion -> list, e.g.
            ``result.flow_history['10b']``.
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the summed flux at each timestep.
    """
    n_steps = len(next(iter(flow_history.values()))[ion])
    return [sum(flow_history[j][ion][t] for j in flow_history) for t in range(n_steps)]


def total_flow_all_pathways(result, ion):
    """Sum flux for one ion across every pathway and every junction in a result.

    Args:
        result: An ``engine.SimulationResult``.
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the summed flux (across all pathways) at each timestep.
    """
    per_pathway = [total_flow(flow_history, ion) for flow_history in result.flow_history.values()]
    return [sum(values) for values in zip(*per_pathway)]


def plot_total_flux_comparison(result_a, result_b, ion='Mg', label_a='Parallel (multi-pathway)', label_b='Averaged', title=None, save_path=None):
    """Compare total flux (summed across all pathways/junctions) between two simulation results.

    Typically used to compare a "parallel claudins" result against an
    "averaged claudin" result for the same scenario, but works for any two
    ``SimulationResult``s sharing a time axis -- pass ``label_a``/``label_b``
    (and optionally ``title``) to relabel the plot for a different comparison
    (e.g. two strand counts of the same scenario).

    Args:
        result_a: The ``SimulationResult`` to plot as the solid line.
        result_b: The ``SimulationResult`` to plot as the dashed line.
        ion: Ion to compare ('Na', 'Cl', or 'Mg').
        label_a: Legend label for ``result_a``.
        label_b: Legend label for ``result_b``.
        title: Plot title; defaults to a "{ion} Reabsorption: {label_a} vs {label_b}" title.
        save_path: If given, save the figure to this path.

    Returns:
        A tuple ``(flux_a, flux_b)`` of the two summed-flux lists.
    """
    flux_a = total_flow_all_pathways(result_a, ion)
    flux_b = total_flow_all_pathways(result_b, ion)

    label = ION_LABELS.get(ion, ion)

    plt.figure(figsize=(10, 4))
    plt.plot(result_a.time_axis, flux_a, label=label_a)
    plt.plot(result_b.time_axis, flux_b, label=label_b, linestyle='--')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.ylabel(f'Total {label} Flow (arbitrary unit)')
    plt.xlabel('Time (a.u.)')
    plt.title(title or f'{label} Reabsorption: {label_a} vs {label_b}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return flux_a, flux_b


def plot_mg_comparison(result_parallel, result_avg, save_path=None):
    """Backwards-compatible alias for ``plot_total_flux_comparison(..., ion='Mg')``."""
    return plot_total_flux_comparison(result_parallel, result_avg, ion='Mg', save_path=save_path)


def terminal_flow(flow_history, ion):
    """One pathway's flux for one ion through its terminal (last) junction only, at each timestep.

    Unlike ``total_flow``, this does not sum across a pathway's junctions: in a
    series chain the same ions pass through every junction in turn, so summing
    them inflates the result by roughly the junction count instead of reporting
    the actual throughput. The terminal junction's flux -- what actually crosses
    into the basolateral reservoir -- is the chain-length-invariant quantity,
    needed when comparing results built from chains of different lengths (e.g.
    different strand counts) rather than two results with the same chain length.

    Args:
        flow_history: dict of junction key -> ion -> list, e.g.
            ``result.flow_history['10b']``, in chain order (as produced by
            ``engine.run_simulation``).
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the terminal junction's flux at each timestep.
    """
    terminal_junction = list(flow_history)[-1]
    return flow_history[terminal_junction][ion]


def terminal_flow_all_pathways(result, ion):
    """Sum each pathway's terminal-junction flux for one ion, across every pathway in a result.

    Pathways are genuinely parallel routes for the same ions, so summing
    across pathways (unlike summing across junctions within one pathway) is
    physically correct -- see ``terminal_flow``.

    Args:
        result: An ``engine.SimulationResult``.
        ion: Ion key, e.g. 'Mg'.

    Returns:
        A list of the summed terminal flux (across all pathways) at each timestep.
    """
    per_pathway = [terminal_flow(flow_history, ion) for flow_history in result.flow_history.values()]
    return [sum(values) for values in zip(*per_pathway)]


def plot_terminal_flux_comparison(result_a, result_b, ion='Mg', label_a='A', label_b='B', title=None, save_path=None):
    """Compare terminal-junction flux (summed across pathways only, not junctions) between two results.

    Use this instead of ``plot_total_flux_comparison`` when ``result_a`` and
    ``result_b`` come from chains of different lengths (e.g. different strand
    counts) -- summing across a chain's junctions is not chain-length-invariant
    (see ``terminal_flow``), so it exaggerates differences between chains of
    different lengths. For two results with the *same* chain length,
    ``plot_total_flux_comparison`` remains the more complete picture (it counts
    every junction, not just the terminal one).

    Args:
        result_a: The ``SimulationResult`` to plot as the solid line.
        result_b: The ``SimulationResult`` to plot as the dashed line.
        ion: Ion to compare ('Na', 'Cl', or 'Mg').
        label_a: Legend label for ``result_a``.
        label_b: Legend label for ``result_b``.
        title: Plot title; defaults to a "{ion} Terminal-Junction Flow: {label_a} vs {label_b}" title.
        save_path: If given, save the figure to this path.

    Returns:
        A tuple ``(flux_a, flux_b)`` of the two terminal-flux lists.
    """
    flux_a = terminal_flow_all_pathways(result_a, ion)
    flux_b = terminal_flow_all_pathways(result_b, ion)

    label = ION_LABELS.get(ion, ion)

    plt.figure(figsize=(10, 4))
    plt.plot(result_a.time_axis, flux_a, label=label_a)
    plt.plot(result_b.time_axis, flux_b, label=label_b, linestyle='--')
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.ylabel(f'Terminal-Junction {label} Flow (arbitrary unit)')
    plt.xlabel('Time (a.u.)')
    plt.title(title or f'{label} Terminal-Junction Flow: {label_a} vs {label_b}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return flux_a, flux_b
