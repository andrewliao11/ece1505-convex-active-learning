from experiment_manager import ExperimentParams, ExperimentManager 

params = ExperimentParams()
params.datatype = "moon"      
params.N = 100         
params.input_dim = 2     
params.labeled_ratio = 0.1
params.sigma = 5             
params.noise = 0.4
params.alpha = 1            
params.K = 2

# Which simulator and learner to use
params.learner = "SVMLearner"
params.simulator = "Simulator"
params.sampler = "RandomSampler"

experiment = ExperimentManager(params)

experiment.run("test")