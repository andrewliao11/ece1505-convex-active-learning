import argparse
from experiment_manager import (
    ExperimentParams, 
    ExperimentManager,
    run_experiments_params_given
)


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--name', default="test", type=str)

parser.add_argument('--datatype', choices=['blob', 'moon'], type=str)
parser.add_argument('--N', default=100, type=int)
parser.add_argument('--input_dim', default=2, type=int)
parser.add_argument('--noise', default=0.4, type=float)
parser.add_argument('--labeled_ratio', default=0.1, type=float)
parser.add_argument('--K', default=10, type=int)

parser.add_argument('--learner', choices=["SVMLearner", "LPLearner"], type=str)
parser.add_argument('--simulator', default="Simulator", type=str)
parser.add_argument('--sampler', choices=["CVXSampler", "RandomSampler", "OptimalSampler"], type=str)

# cvxsampler
parser.add_argument('--sigma', default=5, type=float)
parser.add_argument('--alpha', default=2, type=float)
parser.add_argument('--confidence_type', default="learner", choices=["learner", "perfect_onehot_prob", "perfect_distributional_prob"], type=str)
parser.add_argument('--diversity_type', default="none", choices=["none", "optimal"], type=str)
parser.add_argument('--clustering_type', default="none", choices=["none", "spectral"], type=str)

args = parser.parse_args()


params = ExperimentParams()
params.seed = args.seed
params.name = args.name

params.datatype = args.datatype
params.N = args.N
params.input_dim = args.input_dim
params.labeled_ratio = args.labeled_ratio
params.noise = args.noise
params.K = args.K

params.learner = args.learner
params.simulator = args.simulator
params.sampler = args.sampler

params.confidence_type = args.confidence_type
params.diversity_type = args.diversity_type
params.clustering_type = args.clustering_type
params.sigma = args.sigma
params.alpha = args.alpha


run_experiments_params_given(params)
