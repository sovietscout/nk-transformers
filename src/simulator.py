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
                      cache_dir: str = './results/cache',
                      policy_holdout: str = 'high-phi-pi'):
    rng = np.random.RandomState(seed)
    T_train = T_sim - burn_in

    def sample_params(region: str):
        sigma_val = 1.0 + 2.0 * rng.rand()
        kappa_val = 0.05 + 0.45 * rng.rand()
        if region == 'train':
            phi_pi_val = 1.1 + 1.3 * rng.rand()
        elif region == 'holdout':
            phi_pi_val = 2.4 + 0.6 * rng.rand()
        else:
            phi_pi_val = 1.1 + 1.9 * rng.rand()
        phi_x_val = rng.rand()
        rho_r_val = 0.5 + 0.45 * rng.rand()
        rho_u_val = 0.3 + 0.60 * rng.rand()
        rho_v_val = 0.3 + 0.60 * rng.rand()
        sigma_r_val = 0.005 + 0.025 * rng.rand()
        sigma_u_val = 0.001 + 0.014 * rng.rand()
        sigma_v_val = 0.001 + 0.014 * rng.rand()

        return np.array([
            sigma_val, 0.99, kappa_val, phi_pi_val, phi_x_val,
            rho_r_val, rho_u_val, rho_v_val,
            sigma_r_val, sigma_u_val, sigma_v_val,
        ])

    def region_for_split(split: str):
        if policy_holdout == 'none':
            return 'all'
        if policy_holdout == 'high-phi-pi':
            return 'holdout' if split == 'test' else 'train'
        raise ValueError(f"Unknown policy_holdout: {policy_holdout}")

    def draw_split(count: int, split: str):
        params_out = np.zeros((count, 11), dtype=np.float64)
        obs_out = np.zeros((count, T_train, 3), dtype=np.float64)
        shocks_out = np.zeros((count, T_train, 3), dtype=np.float64)
        draws = 0
        attempts = 0
        max_attempts = count * 10
        region = region_for_split(split)

        print(f'Generating {count} {split} draws (policy region: {region})...')
        while draws < count and attempts < max_attempts:
            attempts += 1
            params = sample_params(region)
            result = simulate_one_draw(params, T_sim, burn_in, rng)
            if result is None:
                continue

            obs, shocks = result
            params_out[draws] = params
            obs_out[draws] = obs
            shocks_out[draws] = shocks
            draws += 1

            if draws % 5000 == 0:
                print(f'  {split}: {draws}/{count} (attempts: {attempts})')

        if draws < count:
            raise RuntimeError(f"Only generated {draws} {split} draws, but need {count}.")

        print(f'Generated {draws} {split} draws in {attempts} attempts')
        return params_out, obs_out, shocks_out

    if n_total != n_train + n_val + n_test:
        print(f'Ignoring n_total={n_total}; using split counts summing to {n_train + n_val + n_test}.')

    train_params, train_obs, train_shocks = draw_split(n_train, 'train')
    val_params, val_obs, val_shocks = draw_split(n_val, 'val')
    test_params, test_obs, test_shocks = draw_split(n_test, 'test')

    # Vectorized construction of X_train
    # train_params: (N, 11) -> (N, T, 11)
    # train_shocks: (N, T, 3)
    # lags: (N, T, 3)
    
    N_train_actual = len(train_params)
    lags_train = np.zeros((N_train_actual, T_train, 3), dtype=np.float32)
    lags_train[:, 1:, :] = train_obs[:, :-1, :]
    
    X_train = np.concatenate([
        np.tile(train_params[:, np.newaxis, :], (1, T_train, 1)),
        train_shocks.astype(np.float32),
        lags_train
    ], axis=2)
    Y_train = train_obs.astype(np.float32)

    X_mean = X_train.reshape(-1, 17).mean(axis=0)
    X_std = X_train.reshape(-1, 17).std(axis=0)
    Y_mean = Y_train.reshape(-1, 3).mean(axis=0)
    Y_std = Y_train.reshape(-1, 3).std(axis=0)

    X_std = np.maximum(X_std, 1e-10)
    Y_std = np.maximum(Y_std, 1e-10)

    stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'Y_mean': Y_mean, 'Y_std': Y_std,
        'policy_holdout': policy_holdout,
    }

    def build_split(params_list, obs_list, shocks_list):
        N = len(params_list)
        lags = np.zeros((N, T_train, 3), dtype=np.float32)
        lags[:, 1:, :] = obs_list[:, :-1, :]
        
        X = np.concatenate([
            np.tile(params_list[:, np.newaxis, :], (1, T_train, 1)),
            shocks_list.astype(np.float32),
            lags
        ], axis=2)
        Y = obs_list.astype(np.float32)
        
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
    with open(cache_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump({
            'policy_holdout': policy_holdout,
            'T_sim': T_sim,
            'burn_in': burn_in,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'seed': seed,
        }, f)

    print(f'Data shapes: train={X_train_norm.shape}, val={X_val_norm.shape}, test={X_test_norm.shape}')
    print(f'Cached to {cache_dir}/')

    return data, stats


def load_and_prepare(cache_dir: str = './results/cache',
                     policy_holdout: str = 'high-phi-pi'):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / 'normalized_data.pkl'
    stats_file = cache_dir / 'norm_stats.pkl'
    raw_file = cache_dir / 'raw_data.pkl'
    metadata_file = cache_dir / 'metadata.pkl'

    if cache_file.exists() and stats_file.exists() and raw_file.exists():
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
        cached_holdout = metadata.get('policy_holdout', 'none')
        if cached_holdout != policy_holdout:
            print(f"Cached data uses policy_holdout={cached_holdout}; regenerating for {policy_holdout}.")
        else:
            print("Loading cached normalised data...")
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
            stats['policy_holdout'] = cached_holdout
            return data, stats

    print("Generating simulation data...")
    data, stats = generate_datasets(cache_dir=str(cache_dir),
                                    policy_holdout=policy_holdout)
    return data, stats


def build_y_only_dataset(data: dict, stats: dict):
    """Build lag-only inputs for the y-only Transformer experiment."""
    y_stats = {
        'X_mean': stats['X_mean'][14:17],
        'X_std': stats['X_std'][14:17],
        'Y_mean': stats['Y_mean'],
        'Y_std': stats['Y_std'],
    }

    def build_split(obs):
        N, T, k = obs.shape
        X = np.zeros((N, T, k), dtype=np.float32)
        for i in range(N):
            for t in range(T):
                X[i, t] = 0.0 if t == 0 else obs[i, t - 1]
        Y = obs.astype(np.float32)
        X = (X - y_stats['X_mean']) / y_stats['X_std']
        Y = (Y - y_stats['Y_mean']) / y_stats['Y_std']
        return X.astype(np.float32), Y.astype(np.float32)

    X_train, Y_train = build_split(data['Y_train_raw'])
    X_val, Y_val = build_split(data['Y_val_raw'])
    X_test, Y_test = build_split(data['Y_test_raw'])

    y_data = {
        'X_train': X_train, 'Y_train': Y_train,
        'X_val': X_val, 'Y_val': Y_val,
        'X_test': X_test, 'Y_test': Y_test,
        'Y_test_raw': data['Y_test_raw'],
        'Y_train_raw': data['Y_train_raw'],
        'Y_val_raw': data['Y_val_raw'],
    }
    return y_data, y_stats


if __name__ == '__main__':
    generate_datasets(n_total=1000, n_train=700, n_val=150, n_test=150)
