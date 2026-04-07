import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

OUTLIER_THRESHOLD = 5.0



def fit_distribution(data: any, dist_name: str) -> tuple:

    if dist_name == 'fixed':
        params = (np.mean(data),)
        generated_values = np.full(len(data), params[0])
        wass_distance = stats.wasserstein_distance(data, generated_values)
        return params, wass_distance, 'fixed'

    dist = getattr(stats, dist_name)
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

    if len(set(data)) == 1:
        if type(data) == list:
            return 'fixed', (data[0],)
        else:
            return 'fixed', (data.iloc[0],)
    if len(data) == 0:
        return 'fixed', (0,)

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
        return np.array([mean_value] * n_sample)

    l = dist.rvs(*params, n_sample)
    l = np.clip(l, min_value, max_value)

    return l



def plot_distribution(data: any, params: tuple, dist: any):

    plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label="Data Histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    y = dist.pdf(x, *params)
    
    plt.plot(x, y, 'r-', lw=2, label=f'Fitted {dist.name} Distribution')
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Fitted {dist.name.capitalize()} Distribution")
    plt.show()


def remove_outliers(data: list, m: float = OUTLIER_THRESHOLD) -> list:

    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    
    return data[s < m].tolist()