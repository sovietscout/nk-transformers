import numpy as np
import pickle
from pathlib import Path


def solve_nk_model(params: np.ndarray):
    sigma, beta, kappa, phi_pi, phi_x, rho_r, rho_u, rho_v = params[:8]

    if sigma <= 0:
        return None

    sigma_inv = 1.0 / sigma

    A = np.array([
        [1.0 + phi_x * sigma_inv, phi_pi * sigma_inv],
        [-kappa, 1.0],
    ])
    B = np.array([
        [1.0, sigma_inv],
        [0.0, beta],
    ])
    C = np.array([
        [sigma_inv, 0.0, -sigma_inv],
        [0.0, 1.0, 0.0],
    ])

    if phi_pi <= 1.0:
        return None

    if np.linalg.matrix_rank(B) < B.shape[0]:
        return None

    Rho = np.diag([rho_r, rho_u, rho_v])
    K = np.kron(np.eye(3), A) - np.kron(Rho.T, B)

    if np.linalg.matrix_rank(K) < K.shape[0]:
        return None

    try:
        P_vec = np.linalg.solve(K, C.ravel())
    except np.linalg.LinAlgError:
        return None

    P = P_vec.reshape(2, 3)
    return P


def simulate_one_draw(params: np.ndarray, T_sim: int, burn_in: int, rng: np.random.RandomState):
    sigma, beta, kappa, phi_pi, phi_x, rho_r, rho_u, rho_v = params[:8]
    sigma_r, sigma_u, sigma_v = params[8:]

    P = solve_nk_model(params)
    if P is None:
        return None

    total_T = T_sim + 1
    eps = rng.randn(total_T, 3) * np.array([sigma_r, sigma_u, sigma_v])

    s = np.zeros((total_T, 3))
    s[0, :] = eps[0, :]
    for t in range(1, total_T):
        s[t, 0] = rho_r * s[t - 1, 0] + eps[t, 0]
        s[t, 1] = rho_u * s[t - 1, 1] + eps[t, 1]
        s[t, 2] = rho_v * s[t - 1, 2] + eps[t, 2]

    y = (P @ s.T).T
    x_sim = y[:, 0:1]
    pi_sim = y[:, 1:2]
    i_sim = phi_pi * pi_sim + phi_x * x_sim + s[:, 2:3]

    T_train = T_sim - burn_in
    observables = np.column_stack([x_sim[burn_in:burn_in + T_train],
                                   pi_sim[burn_in:burn_in + T_train],
                                   i_sim[burn_in:burn_in + T_train]])
    shocks = eps[burn_in:burn_in + T_train, :]

    return observables, shocks


def generate_datasets(n_total: int = 60000,
                      n_train: int = 50000,
                      n_val: int = 5000,
                      n_test: int = 5000,
                      T_sim: int = 200,
                      burn_in: int = 50,
                      seed: int = 42,
                      cache_dir: str = './results/cache'):
    rng = np.random.RandomState(seed)
    T_train = T_sim - burn_in

    all_params = np.zeros((n_total, 11), dtype=np.float64)
    all_obs = np.zeros((n_total, T_train, 3), dtype=np.float64)
    all_shocks = np.zeros((n_total, T_train, 3), dtype=np.float64)

    draws = 0
    attempts = 0
    max_attempts = n_total * 3

    print(f'Generating {n_total} simulation draws from NK model prior...')

    while draws < n_total and attempts < max_attempts:
        attempts += 1

        sigma_val = 1.0 + 2.0 * rng.rand()
        kappa_val = 0.05 + 0.45 * rng.rand()
        phi_pi_val = 1.1 + 1.9 * rng.rand()
        phi_x_val = rng.rand()
        rho_r_val = 0.5 + 0.45 * rng.rand()
        rho_u_val = 0.3 + 0.60 * rng.rand()
        rho_v_val = 0.3 + 0.60 * rng.rand()
        sigma_r_val = 0.005 + 0.025 * rng.rand()
        sigma_u_val = 0.001 + 0.014 * rng.rand()
        sigma_v_val = 0.001 + 0.014 * rng.rand()

        params = np.array([
            sigma_val, 0.99, kappa_val, phi_pi_val, phi_x_val,
            rho_r_val, rho_u_val, rho_v_val,
            sigma_r_val, sigma_u_val, sigma_v_val,
        ])

        result = simulate_one_draw(params, T_sim, burn_in, rng)
        if result is None:
            continue

        obs, shocks = result

        all_params[draws] = params
        all_obs[draws] = obs
        all_shocks[draws] = shocks
        draws += 1

        if draws % 5000 == 0:
            print(f'  {draws}/{n_total} (attempts: {attempts})')

    print(f'Generated {draws} valid draws in {attempts} attempts')

    if draws < n_train + n_val + n_test:
        raise RuntimeError(f"Only generated {draws} draws, but need {n_train + n_val + n_test}. "
                           "Increase max_attempts or adjust prior bounds.")

    train_params = all_params[:n_train]
    train_obs = all_obs[:n_train]
    train_shocks = all_shocks[:n_train]

    val_params = all_params[n_train:n_train + n_val]
    val_obs = all_obs[n_train:n_train + n_val]
    val_shocks = all_shocks[n_train:n_train + n_val]

    test_params = all_params[n_train + n_val:n_train + n_val + n_test]
    test_obs = all_obs[n_train + n_val:n_train + n_val + n_test]
    test_shocks = all_shocks[n_train + n_val:n_train + n_val + n_test]

    X_train = np.zeros((len(train_params), T_train, 17), dtype=np.float32)
    Y_train = train_obs.copy().astype(np.float32)

    for i in range(len(train_params)):
        for t in range(T_train):
            lag = np.zeros(3) if t == 0 else train_obs[i, t - 1, :]
            X_train[i, t, :] = np.concatenate([train_params[i], train_shocks[i, t, :], lag])

    X_mean = X_train.reshape(-1, 17).mean(axis=0)
    X_std = X_train.reshape(-1, 17).std(axis=0)
    Y_mean = Y_train.reshape(-1, 3).mean(axis=0)
    Y_std = Y_train.reshape(-1, 3).std(axis=0)

    X_std = np.maximum(X_std, 1e-10)
    Y_std = np.maximum(Y_std, 1e-10)

    stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std,
    }

    def build_split(params_list, obs_list, shocks_list):
        N = len(params_list)
        X = np.zeros((N, T_train, 17), dtype=np.float32)
        Y = np.zeros((N, T_train, 3), dtype=np.float32)
        for i in range(N):
            for t in range(T_train):
                lag = np.zeros(3) if t == 0 else obs_list[i, t - 1, :]
                X[i, t, :] = np.concatenate([params_list[i], shocks_list[i, t, :], lag])
            Y[i] = obs_list[i]
        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std
        return X.astype(np.float32), Y.astype(np.float32)

    X_train_norm, Y_train_norm = build_split(train_params, train_obs, train_shocks)
    X_val_norm, Y_val_norm = build_split(val_params, val_obs, val_shocks)
    X_test_norm, Y_test_norm = build_split(test_params, test_obs, test_shocks)

    data = {
        'X_train': X_train_norm, 'Y_train': Y_train_norm,
        'X_val': X_val_norm, 'Y_val': Y_val_norm,
        'X_test': X_test_norm, 'Y_test': Y_test_norm,
        'Y_test_raw': test_obs.astype(np.float32),
        'params_test': test_params.astype(np.float32),
        'Y_train_raw': train_obs.astype(np.float32),
        'params_train': train_params.astype(np.float32),
        'Y_val_raw': val_obs.astype(np.float32),
        'params_val': val_params.astype(np.float32),
    }

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / 'normalized_data.pkl', 'wb') as f:
        pickle.dump({
            'X_train': X_train_norm, 'Y_train': Y_train_norm,
            'X_val': X_val_norm, 'Y_val': Y_val_norm,
            'X_test': X_test_norm, 'Y_test': Y_test_norm,
        }, f)
    with open(cache_dir / 'norm_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    with open(cache_dir / 'raw_data.pkl', 'wb') as f:
        pickle.dump({
            'test_obs': test_obs, 'test_params': test_params,
            'train_obs': train_obs, 'train_params': train_params,
            'val_obs': val_obs, 'val_params': val_params,
        }, f)

    print(f'Data shapes: train={X_train_norm.shape}, val={X_val_norm.shape}, test={X_test_norm.shape}')
    print(f'Cached to {cache_dir}/')

    return data, stats


def load_and_prepare(cache_dir: str = './results/cache'):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'normalized_data.pkl'
    stats_file = cache_dir / 'norm_stats.pkl'
    raw_file = cache_dir / 'raw_data.pkl'

    if cache_file.exists() and stats_file.exists() and raw_file.exists():
        print("Loading cached normalized data...")
        with open(cache_file, 'rb') as f:
            norm_data = pickle.load(f)
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        with open(raw_file, 'rb') as f:
            raw_data = pickle.load(f)

        data = dict(norm_data)
        data['Y_test_raw'] = raw_data['test_obs']
        data['params_test'] = raw_data['test_params']
        data['Y_train_raw'] = raw_data['train_obs']
        data['params_train'] = raw_data['train_params']
        data['Y_val_raw'] = raw_data['val_obs']
        data['params_val'] = raw_data['val_params']
        return data, stats

    print("Generating simulation data...")
    data, stats = generate_datasets(cache_dir=str(cache_dir))
    return data, stats


if __name__ == '__main__':
    generate_datasets(n_total=1000, n_train=700, n_val=150, n_test=150)