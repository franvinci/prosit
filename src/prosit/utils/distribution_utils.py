import warnings

import numpy as np
import scipy.stats as stats

OUTLIER_THRESHOLD = 20.0



def fit_distribution(data: any, dist_name: str) -> tuple:

    if dist_name == 'fixed':
        params = (np.mean(data),)
        generated_values = np.full(len(data), params[0])
        wass_distance = stats.wasserstein_distance(data, generated_values)
        return params, wass_distance, 'fixed'

    dist = getattr(stats, dist_name)
    # scipy MLE on heavy-tailed candidates (lognorm, gamma, expon) emits
    # harmless RuntimeWarnings (overflow in exp/divide, divide by zero in
    # log) for some samples. The fit still returns valid parameters, so we
    # silence them locally instead of polluting every run's log.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        params = dist.fit(data)

        # Deterministic Wasserstein: compare empirical quantiles vs theoretical quantiles.
        # This is equivalent to the L1 distance between the two CDFs and avoids
        # the stochastic noise of sampling from the fitted distribution.
        n = len(data)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = dist.ppf(probs, *params)
        theoretical_quantiles = np.clip(theoretical_quantiles, min(data), max(data))
        wass_distance = stats.wasserstein_distance(np.sort(data), theoretical_quantiles)

    return params, wass_distance, dist


def return_best_distribution(data: any, dist_search: list = ['fixed', 'norm', 'expon', 'lognorm', 'uniform']) -> tuple:

    dict_fitting_distributions = dict()
    dict_fitting_dist_params = dict()
    dict_wass = dict()

    if len(data) == 0:
        return 'fixed', (0,)
    if len(set(data)) == 1:
        return 'fixed', (float(np.asarray(data).flat[0]),)

    for dist_name in dist_search:

        params, goodness_of_fit, dist = fit_distribution(data, dist_name)
        dict_fitting_distributions[dist_name] = dist
        dict_wass[dist_name] = goodness_of_fit
        dict_fitting_dist_params[dist_name] = params

    best_fit_dist_name = min(dict_wass, key=dict_wass.get)
    best_fit_dist = dict_fitting_distributions[best_fit_dist_name]
    best_fit_dist_params =  dict_fitting_dist_params[best_fit_dist_name]

    return best_fit_dist, best_fit_dist_params


def sampling_from_dist(
        dist: any, params: tuple,
        min_value: float, max_value: float, mean_value: float,
        n_sample: int = 1000
    ) -> np.array:

    if dist == 'fixed':
        return np.array([mean_value] * n_sample, dtype=float)

    # Heavy-tailed families (e.g. lognorm with large sigma) can emit
    # "overflow encountered in exp" from scipy's _rvs; the resulting
    # +inf values are clipped to max_value below, so the warning is
    # noise. NaNs (from rare underflow paths) are coerced to 0.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        l = dist.rvs(*params, n_sample)
    l = np.nan_to_num(l, nan=0.0, posinf=max_value, neginf=min_value)
    return np.clip(l, min_value, max_value).astype(float)



def remove_outliers(data: list, m: float = OUTLIER_THRESHOLD) -> list:

    data = np.asarray(data)
    # np.median on an empty array emits "Mean of empty slice" / "invalid
    # value in scalar divide"; bail out before touching numpy reductions.
    if data.size == 0:
        return []
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)

    return data[s < m].tolist()