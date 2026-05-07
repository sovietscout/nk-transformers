import numpy as np
from scipy import linalg
from scipy.stats import invwishart


def fit_var_ols(Y: np.ndarray, p: int):
    """Fit VAR(p) to a single multivariable time series by OLS.

    Args:
        Y: (T, k) — time series of k variables
        p:  lag order

    Returns:
        B:    (k, k*p)         — coefficient matrix (excluding intercept)
        c:    (k,)             — intercept vector
        Sigma:(k, k)           — residual covariance matrix
        aic:  float            — AIC value
    """
    T, k = Y.shape
    if T <= p:
        raise ValueError(f"Not enough data: T={T}, p={p}")

    n_obs = T - p
    Z = np.zeros((n_obs, k * p))
    for lag in range(p):
        Z[:, lag * k:(lag + 1) * k] = Y[p - lag - 1:T - lag - 1, :]
    Y_target = Y[p:, :]

    Z_aug = np.column_stack([Z, np.ones(n_obs)])

    try:
        coeffs = linalg.solve(Z_aug.T @ Z_aug, Z_aug.T @ Y_target)
    except linalg.LinAlgError:
        coeffs = linalg.lstsq(Z_aug, Y_target)[0]

    B = coeffs[:k * p, :].T  # (k, k*p)
    c = coeffs[k * p, :]     # (k,)
    residuals = Y_target - Z_aug @ coeffs
    Sigma = (residuals.T @ residuals) / (n_obs - k * p - 1)

    logdet = np.linalg.slogdet(Sigma)[1]
    aic = n_obs * (k + logdet) + 2 * (k * k * p + k)

    return B, c, Sigma, aic


def select_var_order(Y: np.ndarray, p_max: int = 8):
    """Select VAR lag order by AIC.

    Args:
        Y: (T, k) time series
        p_max: maximum lag to consider

    Returns:
        best_p, best_B, best_c, best_Sigma, best_aic
    """
    best_aic = np.inf
    best_result = None
    best_p = 0
    for p in range(1, p_max + 1):
        try:
            B, c, Sigma, aic = fit_var_ols(Y, p)
            if aic < best_aic:
                best_aic = aic
                best_result = (B, c, Sigma)
                best_p = p
        except ValueError:
            break
    return (best_p,) + best_result


def var_forecast(B: np.ndarray, c: np.ndarray, Y_history: np.ndarray,
                 p: int, horizon: int):
    """Generate h-step-ahead VAR forecasts.

    Args:
        B: (k, k*p) coefficients
        c: (k,) intercept
        Y_history: (p, k) most recent p observations (oldest first)
        p: lag order
        horizon: forecast horizon

    Returns:
        (horizon, k) forecast
    """
    k = B.shape[0]
    # Keep the lag vector in the same order used when fitting the VAR.
    lags = Y_history[-p:].flatten()  # (k*p,)
    forecasts = np.zeros((horizon, k))
    for h in range(horizon):
        fc = B @ lags + c
        forecasts[h] = fc
        lags = np.roll(lags, -k)
        lags[-k:] = fc
    return forecasts


def var_irf(B: np.ndarray, Sigma: np.ndarray, p: int,
            horizon: int = 20, shock_index: int = 0,
            shock_size: float = 1.0):
    """Compute IRF from Cholesky-identified VAR.

    Args:
        B:       (k, k*p) coefficient matrix
        Sigma:   (k, k) residual covariance
        p:       lag order
        horizon: IRF horizon
        shock_index: which variable gets the shock (by Cholesky ordering)
        shock_size: magnitude of shock (in Cholesky std dev units)

    Returns:
        (horizon+1, k) IRF for all variables
    """
    k = B.shape[0]
    # Shock the selected Cholesky innovation by one standard deviation.
    L = np.linalg.cholesky(Sigma)
    impact = L[:, shock_index] * shock_size

    irf = np.zeros((horizon + 1, k))
    irf[0, :] = impact

    companion = np.zeros((k * p, k * p))
    companion[:k, :] = B
    if p > 1:
        companion[k:, :k * (p - 1)] = np.eye(k * (p - 1))

    state = np.zeros(k * p)
    state[:k] = impact
    for h in range(1, horizon + 1):
        state = companion @ state
        irf[h, :] = state[:k]

    return irf


def bvar_minnesota_fit(Y: np.ndarray, p: int,
                       lambda1: float = 0.2,
                       lambda2: float = 0.5,
                       lambda3: float = 1.0):
    """Fit BVAR with Minnesota prior (Normal-inverse-Wishart conjugate).

    The prior mean for coefficient on own first lag is 1 (random walk),
    all other coefficients have prior mean 0.

    Args:
        Y:        (T, k) time series
        p:        lag order
        lambda1:  overall tightness
        lambda2:  cross-variable shrinkage
        lambda3:  lag decay

    Returns:
        B_post_mean: (k, k*p+1) posterior mean coefficients (includes intercept)
        Sigma_post_mean: (k, k) posterior mean residual covariance
        S_post: posterior scale matrix for Sigma
        nu_post: posterior df for Sigma
        posterior_samples: list of (B, Sigma) draws (for prediction)
    """
    T, k = Y.shape
    T_eff = T - p
    if T_eff < k + 2:
        raise ValueError(f"Too few observations for BVAR: T_eff={T_eff}")

    Z = np.zeros((T_eff, k * p))
    for lag in range(p):
        Z[:, lag * k:(lag + 1) * k] = Y[p - lag - 1:T - lag - 1, :]
    Y_target = Y[p:, :]

    # Work with demeaned data, then recover the intercept at the end.
    Y_mean = Y_target.mean(axis=0)
    Z_mean = Z.mean(axis=0)
    Y_dm = Y_target - Y_mean
    Z_dm = Z - Z_mean

    # Minnesota prior scaling uses each variable's own AR residual variance.
    sigma_sq = np.zeros(k)
    for i in range(k):
        yi = Y_dm[:, i]
        Xi = np.zeros((T_eff, p))
        for lag_idx in range(p):
            Xi[:, lag_idx] = Y[p - lag_idx - 1:T - lag_idx - 1, i]
        yi_dm = yi
        try:
            bi = linalg.solve(Xi.T @ Xi, Xi.T @ yi_dm)
            resid = yi_dm - Xi @ bi
            sigma_sq[i] = np.var(resid)
        except linalg.LinAlgError:
            sigma_sq[i] = 1.0

    if np.any(sigma_sq < 1e-10):
        sigma_sq = np.maximum(sigma_sq, 1e-10)

    n_coeff = k * p  # per equation
    V_prior = np.zeros((k * n_coeff, k * n_coeff))

    for i in range(k):
        for j in range(k):
            for lag_idx in range(p):
                row = i * n_coeff + lag_idx * k + j
                col = row
                lag_weight = 1.0 / (lag_idx + 1) ** lambda3
                if i == j:
                    if lag_idx == 0:
                        V_prior[row, col] = (lambda1 ** 2) / sigma_sq[i]
                    else:
                        V_prior[row, col] = (lambda1 / lag_weight) ** 2 / sigma_sq[i]
                else:
                    # Cross-variable shrinkage follows the usual Minnesota
                    # scaling by relative residual variance.
                    V_prior[row, col] = (lambda1 * lambda2 / lag_weight) ** 2 * (sigma_sq[i] / sigma_sq[j])

    V_prior_inv = np.diag(1.0 / np.diag(V_prior))

    # Own first lags are centred on a random walk; everything else is zero.
    B_prior = np.zeros((k, n_coeff))
    for i in range(k):
        B_prior[i, i] = 1.0

    S_prior = np.diag(sigma_sq) * (k + 2) * 0.1
    nu_prior = k + 2

    ZtZ = Z_dm.T @ Z_dm
    ZtY = Z_dm.T @ Y_dm

    V_post_inv = ZtZ + V_prior_inv
    try:
        V_post = linalg.inv(V_post_inv)
    except linalg.LinAlgError:
        V_post = linalg.pinv(V_post_inv)

    B_hat = V_post @ (ZtY + V_prior_inv @ B_prior.T)  # (k*p, k)
    B_hat = B_hat.T  # (k, k*p)

    residuals = Y_dm - Z_dm @ B_hat.T
    S_post = S_prior + residuals.T @ residuals + \
             (B_hat - B_prior) @ V_prior_inv @ (B_hat - B_prior).T

    nu_post = nu_prior + T_eff

    Sigma_post_mean = S_post / (nu_post - k - 1) if nu_post > k + 1 else S_post / nu_post

    intercept = Y_mean - B_hat @ Z_mean

    # Very ill-conditioned posteriors make the random draws unreliable; in that
    # case the caller falls back to the posterior mean.
    n_draws = 500
    posterior_draws = []
    rng = np.random.RandomState(42)

    try:
        cond_num = np.linalg.cond(V_post)
        skip_draws = (cond_num > 1e12) or np.isnan(cond_num)
    except Exception:
        skip_draws = True

    if skip_draws:
        pass
    else:
        for _ in range(n_draws):
            try:
                Sigma_draw = invwishart.rvs(df=nu_post, scale=S_post, random_state=rng)
                B_vec_mean = B_hat.T.flatten()
                B_cov = np.kron(Sigma_draw, V_post)
                B_vec_draw = rng.multivariate_normal(B_vec_mean, B_cov)
                B_draw = B_vec_draw.reshape(k, k * p)
                posterior_draws.append((B_draw, Sigma_draw))
            except Exception:
                break

    B_post_mean = np.column_stack([B_hat, intercept])  # (k, k*p+1)

    return {
        'B_post_mean': B_post_mean,
        'Sigma_post_mean': Sigma_post_mean,
        'posterior_draws': posterior_draws,
        'p': p,
        'S_post': S_post,
        'nu_post': nu_post,
    }


def bvar_forecast(bvar_result: dict, Y_history: np.ndarray, horizon: int,
                  n_draws: int = 500):
    """Generate h-step-ahead BVAR forecasts with posterior distribution.

    Args:
        bvar_result: output of bvar_minnesota_fit
        Y_history: (p, k) most recent observations
        horizon: forecast horizon

    Returns:
        fc_mean:  (horizon, k) point forecast (posterior mean)
        fc_lower: (horizon, k) 5th percentile
        fc_upper: (horizon, k) 95th percentile
    """
    k = bvar_result['B_post_mean'].shape[0]
    p = bvar_result['p']
    posterior_draws = bvar_result['posterior_draws']
    B_post_mean = bvar_result['B_post_mean']

    lags_init = Y_history[-p:].flatten()

    # Some small or ill-conditioned fits only have the posterior mean.
    if len(posterior_draws) == 0:
        B_coeff = B_post_mean[:, :-1]  # (k, k*p)
        intercept = B_post_mean[:, -1]  # (k,)
        lags = lags_init.copy()
        fc_mean = np.zeros((horizon, k))
        for h in range(horizon):
            fc = B_coeff @ lags + intercept
            fc_mean[h] = fc
            lags = np.roll(lags, -k)
            lags[-k:] = fc
        fc_lower = fc_mean.copy()
        fc_upper = fc_mean.copy()
    else:
        all_fc = np.zeros((len(posterior_draws), horizon, k))
        for d, (B_draw, _) in enumerate(posterior_draws):
            intercept = B_post_mean[:, -1]
            lags = lags_init.copy()
            for h in range(horizon):
                fc = B_draw @ lags + intercept
                all_fc[d, h] = fc
                lags = np.roll(lags, -k)
                lags[-k:] = fc
        fc_mean = all_fc.mean(axis=0)
        fc_lower = np.percentile(all_fc, 5, axis=0)
        fc_upper = np.percentile(all_fc, 95, axis=0)

    return fc_mean, fc_lower, fc_upper


def bvar_irf(bvar_result: dict, horizon: int, shock_index: int = 0,
             shock_size: float = 1.0):
    """Compute IRF from BVAR posterior.

    Returns:
        irf_mean:  (horizon+1, k)
        irf_lower: (horizon+1, k)
        irf_upper: (horizon+1, k)
    """
    k = bvar_result['B_post_mean'].shape[0]
    p = bvar_result['p']
    posterior_draws = bvar_result['posterior_draws']
    B_post_mean = bvar_result['B_post_mean']

    # Some fits skip posterior draws when the covariance is too unstable.
    if len(posterior_draws) == 0:
        B_mean = B_post_mean[:, :-1]
        try:
            Sigma_mean = bvar_result['Sigma_post_mean']
            L = np.linalg.cholesky(Sigma_mean)
            impact = L[:, shock_index] * shock_size
        except Exception:
            impact = np.zeros(k)

        irf_mean = np.zeros((horizon + 1, k))
        irf_mean[0, :] = impact

        companion = np.zeros((k * p, k * p))
        companion[:k, :] = B_mean
        if p > 1:
            companion[k:, :k * (p - 1)] = np.eye(k * (p - 1))

        state = np.zeros(k * p)
        state[:k] = impact
        for h in range(1, horizon + 1):
            state = companion @ state
            irf_mean[h, :] = state[:k]

        irf_lower = irf_mean.copy()
        irf_upper = irf_mean.copy()
    else:
        all_irf = np.zeros((len(posterior_draws), horizon + 1, k))
        for d, (B_draw, Sigma_draw) in enumerate(posterior_draws):
            L = np.linalg.cholesky(Sigma_draw)
            impact = L[:, shock_index] * shock_size
            all_irf[d, 0, :] = impact

            companion = np.zeros((k * p, k * p))
            companion[:k, :] = B_draw
            if p > 1:
                companion[k:, :k * (p - 1)] = np.eye(k * (p - 1))

            state = np.zeros(k * p)
            state[:k] = impact
            for h in range(1, horizon + 1):
                state = companion @ state
                all_irf[d, h, :] = state[:k]

        irf_mean = all_irf.mean(axis=0)
        irf_lower = np.percentile(all_irf, 5, axis=0)
        irf_upper = np.percentile(all_irf, 95, axis=0)

    return irf_mean, irf_lower, irf_upper


def fit_kalman_var(Y_context: np.ndarray, p: int = 1):
    """Fit a reduced-form VAR(p) represented as a Kalman state model."""
    B, c, Sigma, _ = fit_var_ols(Y_context, p=p)
    k = Y_context.shape[1]
    state_dim = k * p

    A = np.zeros((state_dim, state_dim))
    A[:k, :] = B
    if p > 1:
        A[k:, :k * (p - 1)] = np.eye(k * (p - 1))

    intercept = np.zeros(state_dim)
    intercept[:k] = c
    H = np.zeros((k, state_dim))
    H[:, :k] = np.eye(k)

    Q = np.zeros((state_dim, state_dim))
    Q[:k, :k] = Sigma + np.eye(k) * 1e-8
    R = np.eye(k) * 1e-6

    return {'A': A, 'c': intercept, 'H': H, 'Q': Q, 'R': R, 'p': p}


def kalman_filter_forecast(model: dict, Y: np.ndarray, start_t: int = 50):
    """One-step predictions from a fitted linear Gaussian VAR state model."""
    A = model['A']
    c = model['c']
    H = model['H']
    Q = model['Q']
    R = model['R']
    p = model['p']
    T, k = Y.shape

    preds = np.zeros((T, k))
    preds[:start_t] = Y[:start_t]

    state = Y[start_t - p:start_t].reshape(-1)
    P = np.eye(len(state)) * 1e-4

    for t in range(start_t, T):
        state_pred = A @ state + c
        P_pred = A @ P @ A.T + Q
        y_pred = H @ state_pred
        preds[t] = y_pred

        innovation = Y[t] - y_pred
        S = H @ P_pred @ H.T + R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P_pred @ H.T @ np.linalg.pinv(S)
        state = state_pred + K @ innovation
        P = (np.eye(len(state)) - K @ H) @ P_pred

    return preds
