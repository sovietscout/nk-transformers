"""
plots.py — Generate all figures and tables for the NK Transformer project.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 150,
})

OBS_NAMES = ['Output Gap ($x_t$)', 'Inflation ($\\pi_t$)', 'Interest Rate ($i_t$)']
OBS_SHORT = ['x', 'π', 'i']
COLORS = {
    'True': 'black',
    'Transformer': '#e41a1c',
    'VAR': '#377eb8',
    'BVAR': '#4daf4a',
    'DSGE': '#ff7f00',
}


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: Trajectory overlay (Transformer vs. True)
# ---------------------------------------------------------------------------

def plot_trajectory_overlay(data: dict, stats: dict, model, save_path: str,
                            sim_idx: int = 0, device: str = 'cuda'):
    """Plot true vs. Transformer predicted trajectories for one simulation."""
    import torch
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    X_test = data['X_test']
    Y_test = data['Y_test']

    x_in = torch.from_numpy(X_test[sim_idx]).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model.to(device)(x_in).cpu().numpy()[0]

    true = Y_test[sim_idx]

    # Denormalize
    pred = pred * stats['Y_std'] + stats['Y_mean']
    true = true * stats['Y_std'] + stats['Y_mean']
    T = pred.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, obs_idx, name in zip(axes, range(3), OBS_NAMES):
        ax.plot(range(T), true[:, obs_idx], 'k-', linewidth=1.5, label='True')
        ax.plot(range(T), pred[:, obs_idx], '--', color=COLORS['Transformer'],
                linewidth=1.5, label='Transformer')
        ax.set_xlabel('Quarter')
        ax.set_ylabel(name)
        ax.legend()
        ax.set_xlim(0, T)

    fig.suptitle('Trajectory Overlay: True vs. Transformer', fontsize=14, y=1.02)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: All-model trajectory comparison
# ---------------------------------------------------------------------------

def plot_all_model_trajectories(data: dict, stats: dict, model,
                                var_results: dict, bvar_results: dict,
                                save_path: str, sim_indices: list = [0, 10, 50],
                                device: str = 'cuda'):
    """Plot all four models vs. true for multiple representative simulations."""
    import torch
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    Y_test_raw = data['Y_test_raw']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Get Transformer predictions
    tf_preds = {}
    for idx in sim_indices:
        x_in = torch.from_numpy(X_test[idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model.to(device)(x_in).cpu().numpy()[0]
        tf_preds[idx] = p * stats['Y_std'] + stats['Y_mean']

    fig, axes = plt.subplots(len(sim_indices), 3, figsize=(16, 5 * len(sim_indices)))

    for row, sim_idx in enumerate(sim_indices):
        true = Y_test_raw[sim_idx]
        T = true.shape[0]

        # VAR one-step predictions
        var_pred = np.array(var_results.get('predictions', {}).get(sim_idx,
                                    np.zeros((T, 3))))
        # BVAR one-step predictions
        bvar_pred = np.array(bvar_results.get('predictions', {}).get(sim_idx,
                                      np.zeros((T, 3))))

        for col, (obs_idx, name) in enumerate(zip(range(3), OBS_SHORT)):
            ax = axes[row][col] if len(sim_indices) > 1 else axes[col]
            ax.plot(range(T), true[:, obs_idx], '-', color=COLORS['True'],
                    linewidth=1.5, label='True')
            ax.plot(range(T), tf_preds[sim_idx][:, obs_idx], '--',
                    color=COLORS['Transformer'], linewidth=1, label='Transformer', alpha=0.8)
            if var_pred.any():
                ax.plot(range(T), var_pred[:, obs_idx], ':',
                        color=COLORS['VAR'], linewidth=1, label='VAR', alpha=0.7)
            if bvar_pred.any():
                ax.plot(range(T), bvar_pred[:, obs_idx], '-.',
                        color=COLORS['BVAR'], linewidth=1, label='BVAR', alpha=0.7)

            ax.set_title(f'{name} (Sim {sim_idx})')
            ax.set_xlabel('Quarter')
            if col == 0:
                ax.legend(fontsize=8)

    fig.suptitle('Model Comparison: True vs. All Models', fontsize=14, y=1.01)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: IRF comparison (3x3 grid)
# ---------------------------------------------------------------------------

def plot_irf_grid(irf_results: dict, save_path: str):
    """Plot IRFs for all shock types x observables, comparing all models."""
    shock_labels = {
        'r': 'Natural Rate Shock ($r^n$)',
        'u': 'Cost-Push Shock ($u$)',
        'v': 'Monetary Policy Shock ($v$)',
    }
    var_names = ['x', 'pi', 'i']
    model_names = ['Transformer', 'VAR', 'BVAR']

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    for s_row, (shock_type, shock_label) in enumerate(shock_labels.items()):
        for v_col, v_name in enumerate(var_names):
            ax = axes[s_row][v_col]
            for m_name in model_names:
                if shock_type in irf_results.get(m_name, {}) and \
                   v_name in irf_results[m_name][shock_type]:
                    val = irf_results[m_name][shock_type][v_name]['irf_mse_mean']
                    ax.bar(model_names.index(m_name), val,
                           color=COLORS[m_name], alpha=0.7,
                           label=m_name if v_col == 0 and s_row == 0 else "")
            ax.set_title(f'{shock_label} → {OBS_SHORT[v_col]}')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, fontsize=8)
            ax.set_ylabel('IRF-MSE')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('IRF Accuracy: MSE by Shock Type and Observable', fontsize=14)
    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4: Learning curve (MSE vs. N_training)
# ---------------------------------------------------------------------------

def plot_learning_curve(sample_sizes: list,
                        tf_mse: list, var_mse: list, bvar_mse: list,
                        save_path: str):
    """Plot MSE vs. training sample size (log-log) for all methods."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(sample_sizes, tf_mse, 'o-', color=COLORS['Transformer'],
              linewidth=2, markersize=8, label='Transformer')
    ax.loglog(sample_sizes, var_mse, 's--', color=COLORS['VAR'],
              linewidth=2, markersize=8, label='OLS VAR')
    ax.loglog(sample_sizes, bvar_mse, '^-.', color=COLORS['BVAR'],
              linewidth=2, markersize=8, label='BVAR (Minnesota)')

    ax.set_xlabel('Number of Training Simulations ($N$)')
    ax.set_ylabel('Test MSE')
    ax.legend()
    ax.set_title('Learning Curve: MSE vs. Training Sample Size')
    ax.grid(True, which='both', alpha=0.3)

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 5: MSE vs. forecast horizon
# ---------------------------------------------------------------------------

def plot_forecast_horizon(mses: dict, save_path: str):
    """Plot MSE vs. forecast horizon for all models (semi-log).

    Args:
        mses: dict mapping model_name -> {horizon: mse}
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for m_name, mse_dict in mses.items():
        horizons = sorted(mse_dict.keys())
        values = [mse_dict[h] for h in horizons]
        ax.semilogy(horizons, values, 'o-', color=COLORS.get(m_name, 'gray'),
                    linewidth=2, markersize=8, label=m_name)

    ax.set_xlabel('Forecast Horizon (quarters)')
    ax.set_ylabel('MSE (log scale)')
    ax.legend()
    ax.set_title('Multi-Step Forecast Accuracy')
    ax.grid(True, alpha=0.3)

    ensure_dir(os.path.dirname(save_path))
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_table1(one_step_results: dict):
    """Table 1: One-step-ahead MSE by model and observable."""
    print("\n" + "=" * 80)
    print("TABLE 1: One-Step-Ahead MSE by Model and Observable")
    print("=" * 80)
    header = f"{'Model':<15} {'x':>12} {'π':>12} {'i':>12} {'Overall':>12}"
    print(header)
    print("-" * 63)

    for m_name in ['Transformer', 'VAR', 'BVAR']:
        if m_name in one_step_results:
            overall, per_var = one_step_results[m_name]
            print(f"{m_name:<15} {per_var[0]:>12.6f} {per_var[1]:>12.6f} "
                  f"{per_var[2]:>12.6f} {overall:>12.6f}")
    print()


def print_table2(multistep_results: dict):
    """Table 2: Multi-step-ahead MSE by model and horizon."""
    print("\n" + "=" * 80)
    print("TABLE 2: Multi-Step-Ahead MSE by Model and Horizon")
    print("=" * 80)

    # Collect all horizons
    all_horizons = set()
    for m_res in multistep_results.values():
        all_horizons.update(m_res.keys())
    horizons = sorted(all_horizons)

    header = f"{'Model':<15}"
    for h in horizons:
        header += f" {'h='+str(h):>12}"
    print(header)
    print("-" * (15 + 13 * len(horizons)))

    for m_name in ['Transformer', 'VAR', 'BVAR']:
        if m_name in multistep_results:
            row = f"{m_name:<15}"
            for h in horizons:
                if h in multistep_results[m_name]:
                    row += f" {multistep_results[m_name][h][0]:>12.6f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)
    print()


def print_table3(irf_summary: dict):
    """Table 3: IRF accuracy by model, shock type, observable."""
    print("\n" + "=" * 80)
    print("TABLE 3: IRF Accuracy (MSE / Sign Accuracy)")
    print("=" * 80)

    shock_types = ['r', 'u', 'v']
    var_names = ['x', 'pi', 'i']

    for m_name in ['Transformer', 'VAR', 'BVAR']:
        print(f"\n--- {m_name} ---")
        print(f"{'Shock':<6} {'Var':<4} {'IRF-MSE':>10} {'Sign Acc':>10}")
        print("-" * 32)
        for s in shock_types:
            for v in var_names:
                if m_name in irf_summary and s in irf_summary[m_name] \
                        and v in irf_summary[m_name][s]:
                    entry = irf_summary[m_name][s][v]
                    print(f"{s:<6} {v:<4} {entry['irf_mse_mean']:>10.6f} "
                          f"{entry['sign_acc_mean']:>10.3f}")
