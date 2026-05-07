import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.benchmarks import (
    bvar_forecast,
    bvar_irf,
    bvar_minnesota_fit,
    fit_kalman_var,
    kalman_filter_forecast,
    select_var_order,
    var_forecast,
    var_irf,
)


def unnormalise(Y_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Convert normalised predictions back to original scale."""
    return Y_norm * stats['Y_std'] + stats['Y_mean']


def compute_mse(pred: np.ndarray, true: np.ndarray):
    """Mean squared error, averaged over all dimensions."""
    se = (pred - true) ** 2
    overall = se.mean()
    per_var = se.mean(axis=tuple(range(se.ndim - 1)))
    return overall, per_var


def evaluate_one_step_mse(model, data: dict, stats: dict, device: str = 'cuda'):
    """One-step-ahead MSE for the Transformer on test set."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_test = data['X_test']
    Y_test = data['Y_test']
    N, T, _ = X_test.shape

    all_preds = np.zeros((N, T, 3))
    batch_size = 128
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="One-step MSE"):
            end_i = min(i + batch_size, N)
            x_in = torch.from_numpy(X_test[i:end_i]).to(device)
            pred = model(x_in).cpu().numpy()
            all_preds[i:end_i] = pred

    pred_unnorm = unnormalise(all_preds, stats)
    true_unnorm = unnormalise(Y_test, stats)

    start_t = 50
    overall, per_var = compute_mse(pred_unnorm[:, start_t:, :], true_unnorm[:, start_t:, :])
    return overall, per_var


def _var_one_step_worker(args):
    Y, p_max = args
    T, k = Y.shape
    pred = np.zeros_like(Y)
    try:
        p, B, c, Sigma, _ = select_var_order(Y, p_max=p_max)
        for t in range(p, T):
            lags = Y[t - p:t, :][::-1]
            fc = B @ lags.flatten() + c
            pred[t, :] = fc
        pred[:p, :] = Y[:p, :]
    except Exception:
        pred[:] = Y
    return pred


def evaluate_var_one_step(data: dict, stats: dict):
    """One-step-ahead MSE for OLS VAR."""
    Y_test_raw = data['Y_test_raw']
    N, T, k = Y_test_raw.shape
    p_max = 8

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(_var_one_step_worker, [(Y_test_raw[i], p_max) for i in range(N)]),
                            total=N, desc="VAR one-step MSE"))

    all_preds = np.stack(results)
    start_t = 50
    overall, per_var = compute_mse(all_preds[:, start_t:, :], Y_test_raw[:, start_t:, :])
    return overall, per_var


def _bvar_one_step_worker(args):
    Y = args
    pred = np.zeros_like(Y)
    try:
        bvar_res = bvar_minnesota_fit(Y, p=4, lambda1=0.2, lambda2=0.5, lambda3=1.0)
        B_hat = bvar_res['B_post_mean']
        p = bvar_res['p']
        B_coeff = B_hat[:, :-1]
        intercept = B_hat[:, -1]
        T = Y.shape[0]
        for t in range(p, T):
            lags = Y[t - p:t, :][::-1]
            fc = B_coeff @ lags.flatten() + intercept
            pred[t, :] = fc
        pred[:p, :] = Y[:p, :]
    except Exception:
        pred[:] = Y
    return pred


def evaluate_bvar_one_step(data: dict, stats: dict):
    """One-step-ahead MSE for BVAR."""
    Y_test_raw = data['Y_test_raw']
    N = Y_test_raw.shape[0]

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(_bvar_one_step_worker, [Y_test_raw[i] for i in range(N)]),
                            total=N, desc="BVAR one-step MSE"))

    all_preds = np.stack(results)
    start_t = 50
    overall, per_var = compute_mse(all_preds[:, start_t:, :], Y_test_raw[:, start_t:, :])
    return overall, per_var


def evaluate_multistep_mse(model, data: dict, stats: dict,
                           horizons: list = [1, 4, 8, 12, 20],
                           device: str = 'cuda',
                           batch_size: int = 128):
    """Multi-step-ahead MSE for the Transformer."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    X_test = data['X_test']
    Y_test = data['Y_test']
    N, T, _ = X_test.shape

    results = {}
    for h in horizons:
        all_preds = []
        all_trues = []
        for i in tqdm(range(0, N, batch_size), desc=f"Multi-step h={h}"):
            end_i = min(i + batch_size, N)
            context_end = 50
            if context_end + h > T:
                context_end = T - h
            x_context = torch.from_numpy(X_test[i:end_i, :context_end]).to(device)
            future_shocks = torch.from_numpy(X_test[i:end_i, context_end:context_end + h, 11:14]).to(device)
            
            # Use the true observed lag for the first forecast step to ensure 
            # a fair comparison with the VAR benchmarks.
            true_lag = torch.from_numpy(X_test[i:end_i, context_end, 14:17]).to(device)
            
            pred_norm = model.autoregressive_forecast(x_context, horizon=h, 
                                                      future_shocks=future_shocks,
                                                      true_lag=true_lag)
            all_preds.append(pred_norm.cpu().numpy())
            all_trues.append(Y_test[i:end_i, context_end:context_end + h, :])

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)
        pred_unnorm = unnormalise(all_preds, stats)
        true_unnorm = unnormalise(all_trues, stats)
        overall, per_var = compute_mse(pred_unnorm, true_unnorm)
        results[h] = (overall, per_var)

    return results


def _var_ms_worker(args):
    Y, h = args
    T, k = Y.shape
    try:
        p, B, c, Sigma, _ = select_var_order(Y, p_max=8)
        start_t = max(50, p)
        if start_t + h > T:
            start_t = T - h
        lags = Y[start_t - p:start_t, :]
        fc = var_forecast(B, c, lags, p, h)
        return fc, Y[start_t:start_t + h, :]
    except Exception:
        return np.zeros((h, 3)), np.zeros((h, 3))


def evaluate_var_multistep(data: dict, stats: dict,
                           horizons: list = [1, 4, 8, 12, 20]):
    """Multi-step-ahead MSE for OLS VAR."""
    Y_test_raw = data['Y_test_raw']
    N = Y_test_raw.shape[0]

    results = {}
    for h in horizons:
        with Pool(cpu_count()) as pool:
            res = list(tqdm(pool.imap(_var_ms_worker, [(Y_test_raw[i], h) for i in range(N)]),
                            total=N, desc=f"VAR multi-step h={h}"))
        all_preds = np.stack([r[0] for r in res])
        all_trues = np.stack([r[1] for r in res])
        overall, per_var = compute_mse(all_preds, all_trues)
        results[h] = (overall, per_var)
    return results


def _bvar_ms_worker(args):
    Y, h = args
    T, k = Y.shape
    try:
        bvar_res = bvar_minnesota_fit(Y, p=4)
        p = bvar_res['p']
        start_t = max(50, p)
        if start_t + h > T:
            start_t = T - h
        lags = Y[start_t - p:start_t, :]
        fc_mean, _, _ = bvar_forecast(bvar_res, lags, h, n_draws=200)
        return fc_mean, Y[start_t:start_t + h, :]
    except Exception:
        return np.zeros((h, 3)), np.zeros((h, 3))


def evaluate_bvar_multistep(data: dict, stats: dict,
                            horizons: list = [1, 4, 8, 12, 20]):
    """Multi-step-ahead MSE for BVAR."""
    Y_test_raw = data['Y_test_raw']
    N = Y_test_raw.shape[0]

    results = {}
    for h in horizons:
        with Pool(cpu_count()) as pool:
            res = list(tqdm(pool.imap(_bvar_ms_worker, [(Y_test_raw[i], h) for i in range(N)]),
                            total=N, desc=f"BVAR multi-step h={h}"))
        all_preds = np.stack([r[0] for r in res])
        all_trues = np.stack([r[1] for r in res])
        overall, per_var = compute_mse(all_preds, all_trues)
        results[h] = (overall, per_var)
    return results


def compute_transformer_path_batch(model, params_raw: np.ndarray, shocks_raw: np.ndarray,
                                   stats: dict, device: str = 'cuda'):
    """Simulate batched Transformer paths."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    B, H, _ = shocks_raw.shape
    preds_raw = np.zeros((B, H, 3))
    
    params_norm = (params_raw - stats['X_mean'][:11]) / stats['X_std'][:11]
    shocks_norm = (shocks_raw - stats['X_mean'][11:14]) / stats['X_std'][11:14]
    
    params_norm_t = torch.from_numpy(params_norm).float().to(device).unsqueeze(1)
    shocks_norm_t = torch.from_numpy(shocks_norm).float().to(device)
    
    current_seq = []
    with torch.no_grad():
        for h in range(H):
            if h == 0:
                lag_raw = np.zeros((B, 3))
            else:
                lag_raw = preds_raw[:, h - 1, :]
            
            lag_norm = (lag_raw - stats['X_mean'][14:17]) / stats['X_std'][14:17]
            lag_norm_t = torch.from_numpy(lag_norm).float().to(device).unsqueeze(1)
            
            inp = torch.cat([params_norm_t, shocks_norm_t[:, h:h+1, :], lag_norm_t], dim=-1)
            current_seq.append(inp)
            
            seq_t = torch.cat(current_seq, dim=1)
            pred_norm = model(seq_t)[:, -1, :].cpu().numpy()
            preds_raw[:, h, :] = pred_norm * stats['Y_std'] + stats['Y_mean']
            
    return preds_raw


def compute_transformer_irf_batch(model, params_raw: np.ndarray, shock_vecs: np.ndarray,
                                  stats: dict, horizon: int, device: str = 'cuda'):
    """Compute baseline-subtracted Transformer IRFs for a batch."""
    B = params_raw.shape[0]
    H = horizon + 1
    shocks_shocked = np.zeros((B, H, 3))
    shocks_shocked[:, 0, :] = shock_vecs
    shocks_baseline = np.zeros((B, H, 3))
    
    path_shocked = compute_transformer_path_batch(model, params_raw, shocks_shocked, stats, device=device)
    path_baseline = compute_transformer_path_batch(model, params_raw, shocks_baseline, stats, device=device)
    return path_shocked - path_baseline


def compute_dsge_irf(params_raw: dict, shock_type: str, horizon: int = 20):
    """Compute true analytical IRF from the solved NK model."""
    from src.simulator import solve_nk_model
    params = np.array([
        params_raw['sigma'], params_raw['beta'], params_raw['kappa'], params_raw['phi_pi'], params_raw['phi_x'],
        params_raw['rho_r'], params_raw['rho_u'], params_raw['rho_v'],
        params_raw['sigma_r'], params_raw['sigma_u'], params_raw['sigma_v'],
    ])
    P = solve_nk_model(params)
    if P is None:
        return np.zeros((horizon + 1, 3)), np.zeros(3)
    if shock_type == 'r':
        shock = np.array([params_raw['sigma_r'], 0.0, 0.0])
    elif shock_type == 'u':
        shock = np.array([0.0, params_raw['sigma_u'], 0.0])
    elif shock_type == 'v':
        shock = np.array([0.0, 0.0, params_raw['sigma_v']])
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")
    H = horizon + 1
    s = np.zeros((H, 3))
    s[0] = shock
    for t in range(1, H):
        s[t, 0] = params_raw['rho_r'] * s[t - 1, 0]
        s[t, 1] = params_raw['rho_u'] * s[t - 1, 1]
        s[t, 2] = params_raw['rho_v'] * s[t - 1, 2]
    y = (P @ s.T).T
    x_irf = y[:, 0]
    pi_irf = y[:, 1]
    i_irf = params_raw['phi_pi'] * pi_irf + params_raw['phi_x'] * x_irf + s[:, 2]
    return np.column_stack([x_irf, pi_irf, i_irf]), shock


def _irf_accuracy_worker(args):
    Y_i, params_i, stats, var_names = args
    param_dict = {
        'sigma': params_i[0], 'beta': params_i[1], 'kappa': params_i[2],
        'phi_pi': params_i[3], 'phi_x': params_i[4],
        'rho_r': params_i[5], 'rho_u': params_i[6], 'rho_v': params_i[7],
        'sigma_r': params_i[8], 'sigma_u': params_i[9], 'sigma_v': params_i[10],
    }
    local_results = {m: {s: {v: {'mse': None, 'sign_acc': None} for v in var_names} for s in ['r', 'u', 'v']}
                    for m in ['VAR', 'BVAR']}
    true_irfs = {}
    shock_vecs = {}
    for s_idx, shock_type in enumerate(['r', 'u', 'v']):
        true_irf, shock_vec = compute_dsge_irf(param_dict, shock_type, horizon=20)
        true_irfs[shock_type] = true_irf
        shock_vecs[shock_type] = shock_vec
        try:
            p, B, c, Sigma, _ = select_var_order(Y_i, p_max=8)
            var_irf_result = var_irf(B, Sigma, p, horizon=20, shock_index=s_idx, shock_size=1.0)
        except Exception:
            var_irf_result = np.zeros((21, 3))
        try:
            bvar_res = bvar_minnesota_fit(Y_i, p=4)
            bvar_irf_mean, _, _ = bvar_irf(bvar_res, horizon=20, shock_index=s_idx, shock_size=1.0)
        except Exception:
            bvar_irf_mean = np.zeros((21, 3))
        for m_name, pred_irf in [('VAR', var_irf_result), ('BVAR', bvar_irf_mean)]:
            for v_idx, v_name in enumerate(var_names):
                mse = np.mean((pred_irf[:, v_idx] - true_irf[:, v_idx]) ** 2)
                true_sign = np.sign(true_irf[:, v_idx])
                pred_sign = np.sign(pred_irf[:, v_idx])
                mask = true_sign != 0
                sign_acc = (true_sign[mask] == pred_sign[mask]).mean() if mask.sum() > 0 else 1.0
                local_results[m_name][shock_type][v_name] = {'mse': mse, 'sign_acc': sign_acc}
    return local_results, true_irfs, shock_vecs


def evaluate_irf_accuracy(model, data: dict, stats: dict, n_sims: int = 200, device: str = 'cuda'):
    """Evaluate IRF accuracy for all models."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    Y_test_raw = data['Y_test_raw']
    params_test_raw = data['params_test']
    n_sims = min(n_sims, len(Y_test_raw))
    shock_types = ['r', 'u', 'v']
    var_names = ['x', 'pi', 'i']
    model_names = ['Transformer', 'VAR', 'BVAR']

    with Pool(cpu_count()) as pool:
        benchmark_res = list(tqdm(pool.imap(_irf_accuracy_worker, 
                                            [(Y_test_raw[i], params_test_raw[i], stats, var_names) 
                                             for i in range(n_sims)]),
                                  total=n_sims, desc="IRF evaluation (Benchmarks)"))

    results = {m: {s: {v: {'mse': [], 'sign_acc': []} for v in var_names} for s in shock_types}
               for m in model_names}
    for b_res, _, _ in benchmark_res:
        for m in ['VAR', 'BVAR']:
            for s in shock_types:
                for v in var_names:
                    results[m][s][v]['mse'].append(b_res[m][s][v]['mse'])
                    results[m][s][v]['sign_acc'].append(b_res[m][s][v]['sign_acc'])

    for s_idx, shock_type in enumerate(shock_types):
        all_shock_vecs = np.stack([benchmark_res[i][2][shock_type] for i in range(n_sims)])
        tf_irfs = compute_transformer_irf_batch(model, params_test_raw[:n_sims], all_shock_vecs,
                                                stats, horizon=20, device=str(device))
        for i in range(n_sims):
            true_irf = benchmark_res[i][1][shock_type]
            tf_irf = tf_irfs[i]
            for v_idx, v_name in enumerate(var_names):
                mse = np.mean((tf_irf[:, v_idx] - true_irf[:, v_idx]) ** 2)
                true_sign = np.sign(true_irf[:, v_idx])
                pred_sign = np.sign(tf_irf[:, v_idx])
                mask = true_sign != 0
                sign_acc = (true_sign[mask] == pred_sign[mask]).mean() if mask.sum() > 0 else 1.0
                results['Transformer'][shock_type][v_name]['mse'].append(mse)
                results['Transformer'][shock_type][v_name]['sign_acc'].append(sign_acc)

    summary = {}
    for m_name in model_names:
        summary[m_name] = {}
        for s in shock_types:
            summary[m_name][s] = {}
            for v in var_names:
                summary[m_name][s][v] = {
                    'irf_mse_mean': np.mean(results[m_name][s][v]['mse']),
                    'irf_mse_std': np.std(results[m_name][s][v]['mse']),
                    'sign_acc_mean': np.mean(results[m_name][s][v]['sign_acc']),
                    'sign_acc_std': np.std(results[m_name][s][v]['sign_acc']),
                }
    return summary


def _kalman_worker(args):
    Y, start_t = args
    try:
        kalman = fit_kalman_var(Y[:start_t], p=1)
        return kalman_filter_forecast(kalman, Y, start_t=start_t)
    except Exception:
        return Y


def evaluate_kalman_one_step(data: dict):
    """One-step MSE for a reduced-form Kalman VAR."""
    Y_test_raw = data['Y_test_raw']
    N = Y_test_raw.shape[0]
    start_t = 50
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(_kalman_worker, [(Y_test_raw[i], start_t) for i in range(N)]),
                            total=N, desc="Kalman one-step MSE"))
    all_preds = np.stack(results)
    overall, per_var = compute_mse(all_preds[:, start_t:, :], Y_test_raw[:, start_t:, :])
    return overall, per_var


def collect_irf_paths(model, data: dict, stats: dict, sim_idx: int = 0, horizon: int = 20, device: str = 'cuda'):
    """Return true, Transformer, VAR, and BVAR IRF paths for plotting."""
    Y_i = data['Y_test_raw'][sim_idx]
    params_i = data['params_test'][sim_idx]
    param_dict = {
        'sigma': params_i[0], 'beta': params_i[1], 'kappa': params_i[2],
        'phi_pi': params_i[3], 'phi_x': params_i[4],
        'rho_r': params_i[5], 'rho_u': params_i[6], 'rho_v': params_i[7],
        'sigma_r': params_i[8], 'sigma_u': params_i[9], 'sigma_v': params_i[10],
    }
    paths = {}
    for s_idx, shock_type in enumerate(['r', 'u', 'v']):
        true_irf, shock_vec = compute_dsge_irf(param_dict, shock_type, horizon=horizon)
        # Use single-batch call for Transformer
        tf_irf = compute_transformer_irf_batch(model, params_i[None, :], shock_vec[None, :],
                                               stats, horizon=horizon, device=device)[0]
        try:
            p, B, c, Sigma, _ = select_var_order(Y_i, p_max=8)
            var_path = var_irf(B, Sigma, p, horizon=horizon, shock_index=s_idx, shock_size=1.0)
        except Exception:
            var_path = np.zeros_like(true_irf)
        try:
            bvar_res = bvar_minnesota_fit(Y_i, p=4)
            bvar_path, _, _ = bvar_irf(bvar_res, horizon=horizon, shock_index=s_idx, shock_size=1.0)
        except Exception:
            bvar_path = np.zeros_like(true_irf)
        paths[shock_type] = {'True': true_irf, 'Transformer': tf_irf, 'VAR': var_path, 'BVAR': bvar_path}
    return paths
