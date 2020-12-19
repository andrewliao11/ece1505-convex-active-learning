# convex-active-learning

This is the source code of the final project in ECE1505 Convex Optimization



- To execute the code with `RandomSampler`
```
python run.py --seed SEED --N N_DATA --datatype blob --K N_CLASSES --learner SVMLearner --sampler RandomSampler 
```

- To execute the code with `OptimalSampler`
```
python run.py --seed SEED --N N_DATA --datatype blob --K N_CLASSES --learner SVMLearner --sampler OptimalSampler 
```

- To execute the code with `CVXSampler`
```
python run.py --seed SEED --N N_DATA --datatype blob --K N_CLASSES --learner SVMLearner --sampler CVXSampler 
```
