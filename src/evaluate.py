"""
evaluate.py — Evaluation metrics for all models on the holdout set.
"""

import numpy as np
import torch
from tqdm import tqdm

from src.benchmarks import (select_var_order, var_forecast, var_irf,
                        bvar_minnesota_fit, bvar_forecast, bvar_irf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unnormalize(Y_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Convert normalized predictions back to original scale."""
    return Y_norm * stats['Y_std'] + stats['Y_mean']


def compute_mse(pred: np.ndarray, true: np.ndarray):
    """Mean squared error, averaged over all dimensions.

    Args:
        pred: (N, T, k) or (N, k)
        true: same shape

    Returns:
        float: overall MSE
        np.ndarray: per-variable MSE (k,)
    """
    se = (pred - true) ** 2
    overall = se.mean()
    per_var = se.mean(axis=tuple(range(se.ndim - 1)))  # averages over all but last dim
    return overall, per_var


# ---------------------------------------------------------------------------
# 1. One-step-ahead MSE
# ---------------------------------------------------------------------------

def evaluate_one_step_mse(model, data: dict, stats: dict, device: str = 'cuda'):
    """One-step-ahead MSE for the Transformer on test set.

    The model sees [params, shocks_t, obs_{t-1}] and predicts obs_t.
    This includes the full information set at time t.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_test = data['X_test']
    Y_test = data['Y_test']
    N, T, _ = X_test.shape

    all_preds = np.zeros((N, T, 3))
    with torch.no_grad():
        for i in tqdm(range(N), desc="One-step MSE"):
            # Process full sequence
            x_in = torch.from_numpy(X_test[i]).unsqueeze(0).to(device)  # (1, T, 17)
            pred = model(x_in).cpu().numpy()[0]  # (T, 3)
            all_preds[i] = pred

    pred_unnorm = unnormalize(all_preds, stats)
    true_unnorm = unnormalize(Y_test, stats)

    # MSE over the last 100 steps (after 50-step warm-up)
    start_t = 50
    overall, per_var = compute_mse(pred_unnorm[:, start_t:, :], true_unnorm[:, start_t:, :])
    return overall, per_var


def evaluate_var_one_step(data: dict, stats: dict):
    """One-step-ahead MSE for OLS VAR (one VAR per test simulation)."""
    Y_test_raw = data['Y_test_raw']  # (N, T, 3) raw observables
    N, T, k = Y_test_raw.shape
    p_max = 8

    all_preds = np.zeros((N, T, 3))

    for i in tqdm(range(N), desc="VAR one-step MSE"):
        Y = Y_test_raw[i]  # (T, 3)
        try:
            p, B, c, Sigma, _ = select_var_order(Y, p_max=p_max)
            # Generate one-step forecasts for t = p to T-1
            for t in range(p, T):
                lags = Y[t - p:t, :]
                fc = B @ lags.flatten() + c
                all_preds[i, t, :] = fc
            # For t < p, copy true values
            all_preds[i, :p, :] = Y[:p, :]
        except Exception:
            all_preds[i] = Y  # fallback: predict true

    start_t = 50
    overall, per_var = compute_mse(all_preds[:, start_t:, :], Y_test_raw[:, start_t:, :])
    return overall, per_var


def evaluate_bvar_one_step(data: dict, stats: dict):
    """One-step-ahead MSE for BVAR."""
    Y_test_raw = data['Y_test_raw']
    N, T, k = Y_test_raw.shape

    all_preds = np.zeros((N, T, 3))

    for i in tqdm(range(N), desc="BVAR one-step MSE"):
        Y = Y_test_raw[i]
        try:
            bvar_res = bvar_minnesota_fit(Y, p=4, lambda1=0.2, lambda2=0.5, lambda3=1.0)
            B_hat = bvar_res['B_post_mean']
            p = bvar_res['p']
            B_coeff = B_hat[:, :-1]  # (k, k*p)
            intercept = B_hat[:, -1]  # (k,)

            for t in range(p, T):
                lags = Y[t - p:t, :]
                fc = B_coeff @ lags.flatten() + intercept
                all_preds[i, t, :] = fc
            all_preds[i, :p, :] = Y[:p, :]
        except Exception:
            all_preds[i] = Y

    start_t = 50
    overall, per_var = compute_mse(all_preds[:, start_t:, :], Y_test_raw[:, start_t:, :])
    return overall, per_var


# ---------------------------------------------------------------------------
# 2. Multi-step-ahead MSE
# ---------------------------------------------------------------------------

def evaluate_multistep_mse(model, data: dict, stats: dict,
                           horizons: list = [1, 4, 8, 12, 20],
                           device: str = 'cuda'):
    """Multi-step-ahead MSE for the Transformer.

    For each test simulation, use first 50 steps as context, forecast h steps
    autoregressively.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_test = data['X_test']
    Y_test = data['Y_test']

    results = {}
    for h in horizons:
        all_preds = []
        all_trues = []
        N, T, _ = X_test.shape
        for i in tqdm(range(N), desc=f"Multi-step h={h}"):
            # Use first 50 steps as context; forecast next h steps
            context_end = 50
            if context_end + h > T:
                context_end = T - h

            x_context = torch.from_numpy(X_test[i, :context_end]).unsqueeze(0).to(device)
            future_shocks = torch.from_numpy(X_test[i, context_end:context_end + h, 11:14]).unsqueeze(0).to(device)
            pred_norm = model.autoregressive_forecast(x_context, horizon=h, future_shocks=future_shocks)  # (1, h, 3)
            pred = pred_norm.cpu().numpy()[0]
            true = Y_test[i, context_end:context_end + h, :]

            all_preds.append(pred)
            all_trues.append(true)

        all_preds = np.stack(all_preds)
        all_trues = np.stack(all_trues)
        pred_unnorm = unnormalize(all_preds, stats)
        true_unnorm = unnormalize(all_trues, stats)
        overall, per_var = compute_mse(pred_unnorm, true_unnorm)
        results[h] = (overall, per_var)

    return results


def evaluate_var_multistep(data: dict, stats: dict,
                           horizons: list = [1, 4, 8, 12, 20]):
    """Multi-step-ahead MSE for OLS VAR."""
    Y_test_raw = data['Y_test_raw']
    N, T, k = Y_test_raw.shape

    results = {}
    for h in horizons:
        all_preds = []
        all_trues = []
        for i in tqdm(range(N), desc=f"VAR multi-step h={h}"):
            Y = Y_test_raw[i]
            try:
                p, B, c, Sigma, _ = select_var_order(Y, p_max=8)
                start_t = max(50, p)
                if start_t + h > T:
                    start_t = T - h
                lags = Y[start_t - p:start_t, :]
                fc = var_forecast(B, c, lags, p, h)
                all_preds.append(fc)
                all_trues.append(Y[start_t:start_t + h, :])
            except Exception:
                all_preds.append(np.zeros((h, 3)))
                all_trues.append(np.zeros((h, 3)))

        all_preds = np.stack(all_preds)
        all_trues = np.stack(all_trues)
        overall, per_var = compute_mse(all_preds, all_trues)
        results[h] = (overall, per_var)

    return results


def evaluate_bvar_multistep(data: dict, stats: dict,
                            horizons: list = [1, 4, 8, 12, 20]):
    """Multi-step-ahead MSE for BVAR."""
    Y_test_raw = data['Y_test_raw']
    N, T, k = Y_test_raw.shape

    results = {}
    for h in horizons:
        all_preds = []
        all_trues = []
        for i in tqdm(range(N), desc=f"BVAR multi-step h={h}"):
            Y = Y_test_raw[i]
            try:
                bvar_res = bvar_minnesota_fit(Y, p=4)
                p = bvar_res['p']
                start_t = max(50, p)
                if start_t + h > T:
                    start_t = T - h
                lags = Y[start_t - p:start_t, :]
                fc_mean, _, _ = bvar_forecast(bvar_res, lags, h, n_draws=200)
                all_preds.append(fc_mean)
                all_trues.append(Y[start_t:start_t + h, :])
            except Exception:
                all_preds.append(np.zeros((h, 3)))
                all_trues.append(np.zeros((h, 3)))

        all_preds = np.stack(all_preds)
        all_trues = np.stack(all_trues)
        overall, per_var = compute_mse(all_preds, all_trues)
        results[h] = (overall, per_var)

    return results


# ---------------------------------------------------------------------------
# 3. IRF accuracy
# ---------------------------------------------------------------------------

def compute_transformer_irf(model, params: np.ndarray, shock: np.ndarray,
                            horizon: int, device: str = 'cuda'):
    """Compute IRF from the Transformer for a given shock.

    Accumulates the growing sequence so the Transformer sees all previous
    steps, not just the current single token. This matches how the model
    is used during training (on full-length sequences).

    Args:
        model: trained Transformer
        params: (11,) parameter vector (normalized)
        shock: (3,) shock vector at impact
        horizon: IRF horizon

    Returns:
        (horizon+1, 3) predicted IRF
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Build the full IRF sequence step-by-step, feeding the entire
    # sequence so far (like autoregressive_forecast).
    H = horizon + 1
    all_irf = np.zeros((H, 3))

    # Initialize: at step 0, input = [params, shock, zeros] → output obs_0
    shocks_all = np.zeros((H, 3))
    shocks_all[0] = shock  # only impact period has a shock

    # Build the sequence incrementally
    seq = []
    with torch.no_grad():
        for h in range(H):
            if len(seq) == 0:
                # First step: no lagged observable, use zeros
                lag = np.zeros(3)
            else:
                # Previous prediction is the lag for this step
                lag = all_irf[h - 1, :]

            inp = np.concatenate([params, shocks_all[h], lag]).astype(np.float32)
            seq.append(inp)

            # Stack all previous time steps and run the model
            seq_t = torch.from_numpy(np.stack(seq)).unsqueeze(0).to(device)  # (1, h+1, 17)
            out = model(seq_t).cpu().numpy()[0, -1, :]  # (3,) — predict current time step
            all_irf[h, :] = out

    return all_irf  # (horizon+1, 3)


def compute_dsge_irf(params_raw: dict, shock_type: str, horizon: int = 20):
    """Compute true analytical IRF from the solved NK model.

    Uses the policy matrix P from solve_nk_model to compute IRF
    via forward simulation with one-time shock at t=0.
    """
    from src.simulator import solve_nk_model

    sigma = params_raw['sigma']
    beta = params_raw['beta']
    kappa = params_raw['kappa']
    phi_pi = params_raw['phi_pi']
    phi_x = params_raw['phi_x']
    rho_r = params_raw['rho_r']
    rho_u = params_raw['rho_u']
    rho_v = params_raw['rho_v']
    sigma_r = params_raw['sigma_r']
    sigma_u = params_raw['sigma_u']
    sigma_v = params_raw['sigma_v']

    params = np.array([
        sigma, beta, kappa, phi_pi, phi_x,
        rho_r, rho_u, rho_v,
        sigma_r, sigma_u, sigma_v,
    ])

    P = solve_nk_model(params)
    if P is None:
        return np.zeros((horizon + 1, 3)), np.zeros(3)

    if shock_type == 'r':
        shock = np.array([sigma_r, 0.0, 0.0])
    elif shock_type == 'u':
        shock = np.array([0.0, sigma_u, 0.0])
    elif shock_type == 'v':
        shock = np.array([0.0, 0.0, sigma_v])
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")

    H = horizon + 1
    s = np.zeros((H, 3))
    s[0] = shock
    for t in range(1, H):
        s[t, 0] = rho_r * s[t - 1, 0]
        s[t, 1] = rho_u * s[t - 1, 1]
        s[t, 2] = rho_v * s[t - 1, 2]

    y = (P @ s.T).T
    x_irf = y[:, 0]
    pi_irf = y[:, 1]
    i_irf = phi_pi * pi_irf + phi_x * x_irf + s[:, 2]

    irf = np.column_stack([x_irf, pi_irf, i_irf])
    return irf, shock


def evaluate_irf_accuracy(model, data: dict, stats: dict,
                          n_sims: int = 200, device: str = 'cuda'):
    """Evaluate IRF accuracy for all models on a subset of test simulations.

    Returns:
        dict with IRF-MSE and sign accuracy for each model, shock type, observable.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # We need raw (unnormalized) data for VAR/BVAR IRF and true IRF
    Y_test_raw = data['Y_test_raw']
    params_test_raw = data['params_test']

    n_sims = min(n_sims, len(Y_test_raw))

    model_names = ['Transformer', 'VAR', 'BVAR']
    shock_types = ['r', 'u', 'v']
    var_names = ['x', 'pi', 'i']

    # Store: model -> shock -> observable -> list of (irf_mse, sign_acc)
    results = {m: {s: {v: {'mse': [], 'sign_acc': []} for v in var_names} for s in shock_types}
               for m in model_names}

    for i in tqdm(range(n_sims), desc="IRF evaluation"):
        Y_i = Y_test_raw[i]
        params_i = params_test_raw[i]

        param_dict = {
            'sigma': params_i[0], 'beta': params_i[1], 'kappa': params_i[2],
            'phi_pi': params_i[3], 'phi_x': params_i[4],
            'rho_r': params_i[5], 'rho_u': params_i[6], 'rho_v': params_i[7],
            'sigma_r': params_i[8], 'sigma_u': params_i[9], 'sigma_v': params_i[10],
        }

        # Normalized params for Transformer
        params_norm = (params_i - stats['X_mean'][:11]) / stats['X_std'][:11]

        for s_idx, shock_type in enumerate(shock_types):
            # True IRF (structural, from DSGE model)
            true_irf, shock_vec = compute_dsge_irf(param_dict, shock_type, horizon=20)
            true_irf_unnorm = true_irf  # already in original scale

            # VAR/BVAR IRF: Note these use Cholesky identification which imposes
            # a recursive ordering (monetary→inflation→output). This is arbitrary
            # and has no economic basis relative to true structural shocks.
            # Comparing Cholesky IRFs to true structural IRFs is therefore
            # apples-to-oranges at the theoretical level.

            # VAR IRF
            try:
                p, B, c, Sigma, _ = select_var_order(Y_i, p_max=8)
                var_irf_result = var_irf(B, Sigma, p, horizon=20,
                                         shock_index=s_idx, shock_size=1.0)
            except Exception:
                var_irf_result = np.zeros((21, 3))

            # BVAR IRF
            try:
                bvar_res = bvar_minnesota_fit(Y_i, p=4)
                bvar_irf_mean, _, _ = bvar_irf(bvar_res, horizon=20,
                                               shock_index=s_idx, shock_size=1.0)
            except Exception:
                bvar_irf_mean = np.zeros((21, 3))

            # Transformer IRF
            shock_norm = shock_vec / stats['X_std'][11:14]  # normalize shock
            tf_irf_norm = compute_transformer_irf(model, params_norm, shock_norm,
                                                  horizon=20, device=str(device))
            tf_irf_unnorm = tf_irf_norm * stats['Y_std'] + stats['Y_mean']

            model_irfs = {
                'Transformer': tf_irf_unnorm,
                'VAR': var_irf_result,
                'BVAR': bvar_irf_mean,
            }

            for m_name in model_names:
                pred_irf = model_irfs[m_name]
                for v_idx, v_name in enumerate(var_names):
                    # IRF-MSE
                    mse = np.mean((pred_irf[:, v_idx] - true_irf_unnorm[:, v_idx]) ** 2)
                    results[m_name][shock_type][v_name]['mse'].append(mse)

                    # Sign accuracy
                    true_sign = np.sign(true_irf_unnorm[:, v_idx])
                    pred_sign = np.sign(pred_irf[:, v_idx])
                    # Only count where true sign is non-zero
                    mask = true_sign != 0
                    if mask.sum() > 0:
                        sign_acc = (true_sign[mask] == pred_sign[mask]).mean()
                    else:
                        sign_acc = 1.0
                    results[m_name][shock_type][v_name]['sign_acc'].append(sign_acc)

    # Aggregate
    summary = {}
    for m_name in model_names:
        summary[m_name] = {}
        for s in shock_types:
            summary[m_name][s] = {}
            for v in var_names:
                ms = results[m_name][s][v]['mse']
                sa = results[m_name][s][v]['sign_acc']
                summary[m_name][s][v] = {
                    'irf_mse_mean': np.mean(ms),
                    'irf_mse_std': np.std(ms),
                    'sign_acc_mean': np.mean(sa),
                    'sign_acc_std': np.std(sa),
                }

    return summary
