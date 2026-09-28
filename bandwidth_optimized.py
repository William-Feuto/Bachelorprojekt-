import numpy as np
from sklearn.neighbors import KernelDensity
from pathlib import Path
import joblib
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
import hydra
from omegaconf import DictConfig


def loocv_single(index, error_vectors, bandwidth):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    training_data = np.delete(error_vectors, index, axis=0)
    test_sample = error_vectors[index].reshape(1, -1)
    kde.fit(training_data)
    return kde.score_samples(test_sample)[0]


def compute_loocv_log_likelihood(error_vectors, bandwidth, n_jobs=-1):
    window_count = error_vectors.shape[0]
    scores = Parallel(n_jobs=n_jobs)(
        delayed(loocv_single)(i, error_vectors, bandwidth)
        for i in range(window_count)
    )
    return np.sum(scores)


def optimize_bandwidth(error_vectors, min_bandwidth, max_bandwidth):
    tested_bandwidths = []
    tested_log_likelihoods = []

    def objective(bw):
        log_likelihood = compute_loocv_log_likelihood(error_vectors, bw)
        tested_bandwidths.append(bw)
        tested_log_likelihoods.append(log_likelihood)
        return -log_likelihood

    result = minimize_scalar(
        objective,
        bounds=(min_bandwidth, max_bandwidth),
        method="bounded",
        options={"xatol": 1e-4},
    )

    return result.x, tested_bandwidths, tested_log_likelihoods


@hydra.main(version_base=None, config_path="config", config_name="kde")
def main(cfg: DictConfig):

    model_root = Path(cfg.kde.models_root)
    model_name = cfg.kde.model_name
    window_size = cfg.kde.window_size

    model_folder = model_root / model_name / f"window_size_{window_size}"
    error_file = model_folder / "all_mse.npy"

    error_vectors = np.load(error_file)

    min_bandwidth = cfg.kde.min_bandwidth
    max_bandwidth = cfg.kde.max_bandwidth

    optimal_bw, bw_history, ll_history = optimize_bandwidth(
        error_vectors,
        min_bandwidth,
        max_bandwidth
    )

    np.save(model_folder / "kde_optimal_bandwidth.npy", np.array([optimal_bw]))
    np.save(model_folder / "kde_tested_bandwidths.npy", np.array(bw_history))
    np.save(model_folder / "kde_log_likelihood_history.npy", np.array(ll_history))

    print("Optimized LOOCV bandwidth:", optimal_bw)

    kde = KernelDensity(kernel="gaussian", bandwidth=float(optimal_bw))
    kde.fit(error_vectors)

    log_densities = kde.score_samples(error_vectors)
    densities = np.exp(log_densities)

    threshold = np.percentile(densities, 0)
    print(f"Anomaly threshold: {threshold}")

    np.save(model_folder / "kde_train_densities.npy", densities)
    np.save(model_folder / "kde_threshold.npy", np.array([threshold]))

    joblib.dump(kde, model_folder / "kde_model.pkl")

    print(f"KDE model saved in: {model_folder}")


if __name__ == "__main__":
    main()

