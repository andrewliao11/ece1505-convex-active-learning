from experiment_manager import (
    ExperimentParams, 
    ExperimentManager,
    run_experiments,
    compare_experiments
)


experiments_to_compare = [
    "blob_randomsampler", 
    "blob_cvxsampler_argmax",
]

compare_experiments(experiments_to_compare, "temp2.pdf")

'''
experiments_to_compare = [
    "blob_cvxsampler_argmax",
    "blob_cvxsampler_spectral"
]
compare_experiments(experiments_to_compare, "blob_cvx_clustering.pdf")

experiments_to_compare = [
    "moon_cvxsampler_argmax",
    "moon_cvxsampler_spectral"
]
compare_experiments(experiments_to_compare, "moon_cvx_clustering.pdf")

experiments_to_compare = [
    "blob_randomsampler", 
    "blob_cvxsampler", 
]
compare_experiments(experiments_to_compare, "blob_cvx_vs_random.pdf")

experiments_to_compare = [
    "blob_cvxsampler", 
    "blob_cvxsampler_no_conf", 
    "blob_cvxsampler_perf_conf", 
    "blob_cvxsampler_optimal_diversity",
    "blob_optimalsampler",
]
compare_experiments(experiments_to_compare, "blob_cvx.pdf")

'''
