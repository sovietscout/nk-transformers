from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'axes.titleweight': 'semibold',
    'axes.edgecolor': '#9aa0a6',
    'axes.linewidth': 0.8,
    'legend.fontsize': 12,
    'figure.dpi': 320,
    'savefig.bbox': 'tight',
    'savefig.dpi': 440,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'grid.color': '#d8dde3',
    'grid.linewidth': 0.8,
})

OBS_NAMES = ['Output Gap ($x_t$)', 'Inflation ($\\pi_t$)', 'Interest Rate ($i_t$)']
OBS_SHORT = ['x', 'π', 'i']
COLOURS = {
    'True': '#111827',
    'Transformer': '#b42318',
    'VAR': '#2563eb',
    'BVAR': '#0f766e',
    'Kalman VAR': '#7c3aed',
    'DSGE': '#d97706',
}
LINESTYLES = {
    'True': '-',
    'Transformer': '--',
    'VAR': ':',
    'BVAR': '-.',
    'Kalman VAR': (0, (5, 2, 1.5, 2)),
}
LINEWIDTHS = {
    'True': 1.3,
    'Transformer': 1.15,
    'VAR': 1.0,
    'BVAR': 1.0,
    'Kalman VAR': 1.0,
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_axis(ax, xlabel=None, ylabel=None):
    ax.grid(True, axis='y', alpha=0.55)
    ax.grid(True, axis='x', alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9aa0a6')
    ax.spines['bottom'].set_color('#9aa0a6')
    ax.tick_params(axis='both', colors='#374151', labelsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def add_shared_legend(fig, axes, ncol=4):
    first_ax = np.ravel(axes)[0]
    handles, labels = first_ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=ncol,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))


def save_fig(fig, save_path):
    ensure_dir(Path(save_path).parent)
    fig.savefig(save_path)
    plt.close(fig)


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

    pred = pred * stats['Y_std'] + stats['Y_mean']
    true = true * stats['Y_std'] + stats['Y_mean']
    T = pred.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, obs_idx, name in zip(axes, range(3), OBS_NAMES):
        ax.plot(range(T), true[:, obs_idx], color=COLOURS['True'],
                linestyle=LINESTYLES['True'], linewidth=LINEWIDTHS['True'], label='True')
        ax.plot(range(T), pred[:, obs_idx], color=COLOURS['Transformer'],
                linestyle=LINESTYLES['Transformer'], linewidth=LINEWIDTHS['Transformer'],
                label='Transformer')
        format_axis(ax, xlabel='Quarter', ylabel=name)
        ax.set_xlim(0, T)
        ax.margins(x=0.01, y=0.16)

    add_shared_legend(fig, axes, ncol=2)
    fig.suptitle('Trajectory Overlay: True vs. Transformer', y=1.10)
    save_fig(fig, save_path)


def plot_all_model_trajectories(data: dict, stats: dict, model,
                                var_results: dict, bvar_results: dict,
                                save_path: str, sim_indices: list = [0, 10, 50],
                                device: str = 'cuda'):
    """Plot all four models vs. true for multiple representative simulations."""
    import torch
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    Y_test_raw = data['Y_test_raw']
    X_test = data['X_test']

    tf_preds = {}
    for idx in sim_indices:
        x_in = torch.from_numpy(X_test[idx]).unsqueeze(0).to(device)
        with torch.no_grad():
            p = model.to(device)(x_in).cpu().numpy()[0]
        tf_preds[idx] = p * stats['Y_std'] + stats['Y_mean']

    fig, axes = plt.subplots(len(sim_indices), 3, figsize=(18, 5.4 * len(sim_indices)),
                             constrained_layout=True)

    for row, sim_idx in enumerate(sim_indices):
        true = Y_test_raw[sim_idx]
        T = true.shape[0]

        var_pred = np.array(var_results.get('predictions', {}).get(sim_idx,
                                    np.zeros((T, 3))))
        bvar_pred = np.array(bvar_results.get('predictions', {}).get(sim_idx,
                                      np.zeros((T, 3))))

        for col, (obs_idx, name) in enumerate(zip(range(3), OBS_SHORT)):
            ax = axes[row][col] if len(sim_indices) > 1 else axes[col]
            ax.plot(range(T), true[:, obs_idx], linestyle=LINESTYLES['True'],
                    color=COLOURS['True'], linewidth=LINEWIDTHS['True'], label='True')
            ax.plot(range(T), tf_preds[sim_idx][:, obs_idx],
                    linestyle=LINESTYLES['Transformer'], color=COLOURS['Transformer'],
                    linewidth=LINEWIDTHS['Transformer'], label='Transformer')
            if var_pred.any():
                ax.plot(range(T), var_pred[:, obs_idx], linestyle=LINESTYLES['VAR'],
                        color=COLOURS['VAR'], linewidth=LINEWIDTHS['VAR'], label='VAR')
            if bvar_pred.any():
                ax.plot(range(T), bvar_pred[:, obs_idx], linestyle=LINESTYLES['BVAR'],
                        color=COLOURS['BVAR'], linewidth=LINEWIDTHS['BVAR'], label='BVAR')

            ax.set_title(f'{name} (Sim {sim_idx})')
            format_axis(ax, xlabel='Quarter', ylabel='Response' if col == 0 else None)
            ax.margins(x=0.01, y=0.16)

    add_shared_legend(fig, axes)
    fig.suptitle('Model Comparison: True vs. Predictive Models', y=1.04)
    save_fig(fig, save_path)


def plot_reduced_form_trajectory_overlay(data: dict, stats: dict, model,
                                         save_path: str, sim_idx: int = 0,
                                         device: str = 'cuda'):
    """Plot true trajectory against Transformer, VAR, and BVAR one-step fits."""
    import torch
    from src.benchmarks import select_var_order, bvar_minnesota_fit

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    Y = data['Y_test_raw'][sim_idx]
    X = data['X_test'][sim_idx]
    T = Y.shape[0]

    x_in = torch.from_numpy(X).unsqueeze(0).to(device)
    with torch.no_grad():
        tf_pred = model.to(device)(x_in).cpu().numpy()[0]
    tf_pred = tf_pred * stats['Y_std'] + stats['Y_mean']

    try:
        p, B, c, _, _ = select_var_order(Y, p_max=8)
        var_pred = np.zeros_like(Y)
        var_pred[:p] = Y[:p]
        for t in range(p, T):
            var_pred[t] = B @ Y[t - p:t].flatten() + c
    except Exception:
        var_pred = np.full_like(Y, np.nan)

    try:
        bvar_res = bvar_minnesota_fit(Y, p=4)
        p = bvar_res['p']
        B_hat = bvar_res['B_post_mean']
        bvar_pred = np.zeros_like(Y)
        bvar_pred[:p] = Y[:p]
        for t in range(p, T):
            bvar_pred[t] = B_hat[:, :-1] @ Y[t - p:t].flatten() + B_hat[:, -1]
    except Exception:
        bvar_pred = np.full_like(Y, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    for ax, obs_idx, name in zip(axes, range(3), OBS_NAMES):
        ax.plot(range(T), Y[:, obs_idx], linestyle=LINESTYLES['True'],
                color=COLOURS['True'], linewidth=LINEWIDTHS['True'], label='True')
        ax.plot(range(T), tf_pred[:, obs_idx], linestyle=LINESTYLES['Transformer'],
                color=COLOURS['Transformer'], linewidth=LINEWIDTHS['Transformer'],
                label='Transformer')
        ax.plot(range(T), var_pred[:, obs_idx], linestyle=LINESTYLES['VAR'],
                color=COLOURS['VAR'], linewidth=LINEWIDTHS['VAR'], label='VAR')
        ax.plot(range(T), bvar_pred[:, obs_idx], linestyle=LINESTYLES['BVAR'],
                color=COLOURS['BVAR'], linewidth=LINEWIDTHS['BVAR'], label='BVAR')
        format_axis(ax, xlabel='Quarter', ylabel=name)
        ax.set_xlim(0, T)
        ax.margins(x=0.01, y=0.16)

    add_shared_legend(fig, axes)
    fig.suptitle('Trajectory Overlay: True vs. Predictive Models', y=1.10)
    save_fig(fig, save_path)


def plot_irf_grid(irf_results: dict, save_path: str):
    """Plot IRFs for all shock types x observables, comparing all models."""
    shock_labels = {
        'r': 'Natural Rate Shock ($r^n$)',
        'u': 'Cost-Push Shock ($u$)',
        'v': 'Monetary Policy Shock ($v$)',
    }
    var_names = ['x', 'pi', 'i']
    model_names = ['Transformer', 'VAR', 'BVAR']

    fig, axes = plt.subplots(3, 3, figsize=(17, 14), constrained_layout=True)

    for s_row, (shock_type, shock_label) in enumerate(shock_labels.items()):
        for v_col, v_name in enumerate(var_names):
            ax = axes[s_row][v_col]
            vals = []
            for m_name in model_names:
                if shock_type in irf_results.get(m_name, {}) and \
                   v_name in irf_results[m_name][shock_type]:
                    val = irf_results[m_name][shock_type][v_name]['irf_mse_mean']
                    vals.append(val)
                    ax.bar(model_names.index(m_name), val,
                           color=COLOURS[m_name], alpha=0.88, width=0.62,
                           label=m_name if v_col == 0 and s_row == 0 else "")
            ax.set_title(f'{shock_label} -> {OBS_SHORT[v_col]}')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=20, ha='right')
            if vals:
                ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1)
            format_axis(ax, ylabel='IRF MSE')

    add_shared_legend(fig, axes, ncol=3)
    fig.suptitle('Impulse Response Accuracy: MSE by Shock and Observable', y=1.03)
    save_fig(fig, save_path)


def plot_irf_paths(irf_paths: dict, save_path: str):
    """Plot article-style IRF paths for true DSGE and fitted models."""
    shock_labels = {
        'r': 'Natural Rate Shock ($r^n$)',
        'u': 'Cost-Push Shock ($u$)',
        'v': 'Monetary Policy Shock ($v$)',
    }
    fig, axes = plt.subplots(3, 3, figsize=(17, 13.5), sharex=True,
                             constrained_layout=True)
    for row, (shock_type, shock_label) in enumerate(shock_labels.items()):
        for col, obs_name in enumerate(OBS_SHORT):
            ax = axes[row][col]
            for model_name, path in irf_paths.get(shock_type, {}).items():
                ax.plot(path[:, col], linestyle=LINESTYLES.get(model_name, '-'),
                        color=COLOURS.get(model_name, 'grey'),
                        linewidth=LINEWIDTHS.get(model_name, 2.0),
                        label=model_name if row == 0 and col == 0 else None)
            ax.axhline(0, color='#6b7280', linewidth=0.9, alpha=0.55)
            ax.set_title(f'{shock_label} -> {obs_name}')
            format_axis(ax, xlabel='Quarter' if row == 2 else None,
                        ylabel='Response' if col == 0 else None)
            ax.margins(x=0.02, y=0.18)

    add_shared_legend(fig, axes)
    fig.suptitle('Impulse Response Paths', y=1.03)
    save_fig(fig, save_path)


def plot_learning_curve(sample_sizes: list,
                        tf_mse: list, var_mse: list, bvar_mse: list,
                        save_path: str):
    """Plot MSE vs. training sample size (log-log) for all methods."""
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)

    ax.loglog(sample_sizes, tf_mse, 'o-', color=COLOURS['Transformer'],
              linewidth=LINEWIDTHS['Transformer'], markersize=8, label='Transformer')
    ax.loglog(sample_sizes, var_mse, 's--', color=COLOURS['VAR'],
              linewidth=LINEWIDTHS['VAR'], markersize=8, label='OLS VAR')
    ax.loglog(sample_sizes, bvar_mse, '^-.', color=COLOURS['BVAR'],
              linewidth=LINEWIDTHS['BVAR'], markersize=8, label='BVAR (Minnesota)')

    format_axis(ax, xlabel='Number of training simulations ($N$)', ylabel='Test MSE')
    ax.legend(frameon=False, loc='best')
    ax.set_title('Learning Curve: MSE vs. Training Sample Size')
    ax.margins(x=0.04, y=0.18)

    save_fig(fig, save_path)


def plot_forecast_horizon(mses: dict, save_path: str):
    """Plot MSE vs. forecast horizon for all models (semi-log).

    Args:
        mses: dict mapping model_name -> {horizon: mse}
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.2), constrained_layout=True)

    for m_name, mse_dict in mses.items():
        horizons = sorted(mse_dict.keys())
        values = [mse_dict[h][0] if isinstance(mse_dict[h], tuple) else mse_dict[h]
                  for h in horizons]
        ax.semilogy(horizons, values, marker='o',
                    linestyle=LINESTYLES.get(m_name, '-'),
                    color=COLOURS.get(m_name, 'grey'),
                    linewidth=LINEWIDTHS.get(m_name, 2.0),
                    markersize=8, label=m_name)

    format_axis(ax, xlabel='Forecast horizon (quarters)', ylabel='MSE (log scale)')
    ax.legend(frameon=False, loc='best')
    ax.set_title('Multi-Step Forecast Accuracy')
    ax.margins(x=0.05, y=0.18)

    save_fig(fig, save_path)

