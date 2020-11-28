from experiment_manager import (
    ExperimentParams, 
    ExperimentManager,
    run_experiments,
    compare_experiments
)

experiments_to_compare = [
    "blob_cvxsampler_argmax",
    "blob_cvxsampler_spectral"
]
compare_experiments(experiments_to_compare, "blob_cvx_clustering.jpeg")

experiments_to_compare = [
    "blob_randomsampler", 
    "blob_cvxsampler", 
    "blob_cvxsampler_no_conf", 
    "blob_cvxsampler_perf_conf", 
    "blob_optimal_sampler", 
    "blob_cvxsampler_optimal_diversity"
]
compare_experiments(experiments_to_compare, "blob.jpeg")

