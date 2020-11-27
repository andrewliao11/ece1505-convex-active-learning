from experiment_manager import (
    ExperimentParams, 
    ExperimentManager,
    run_experiments
)

# experiments_to_run = ["blob_randomsampler", "blob_cvxsampler", "blob_cvxsampler_no_conf", "blob_cvxsampler_perf_conf"]
experiments_to_run = ["blob_optimalsampler"]
run_experiments(experiments_to_run)

