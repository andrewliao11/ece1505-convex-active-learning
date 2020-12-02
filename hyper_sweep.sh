learner="SVMLearner"
N=100
for seed in 1 2 3 4 5; do
    for datatype in "blob"; do
        for K in 5 10 20; do

        
            sampler="RandomSampler"
            name=seed_$seed-N_$N-datatype_$datatype-K_$K-learner_$learner-sampler_$sampler
            bash launch_slurm_job_on_cpu.sh $name "python run.py --name $name --seed $seed --N $N --datatype $datatype --K $K --learner $learner --sampler $sampler"
            
            sampler="OptimalSampler"
            name=seed_$seed-N_$N-datatype_$datatype-K_$K-learner_$learner-sampler_$sampler
            bash launch_slurm_job_on_cpu.sh $name "python run.py --name $name --seed $seed --N $N --datatype $datatype --K $K --learner $learner --sampler $sampler"
            
            sampler="CVXSampler"
            basename=seed_$seed-N_$N-datatype_$datatype-K_$K-learner_$learner-sampler_$sampler
            for sigma in 3 5; do
                for alpha in 1 2 3; do
                    for confidence_type in "learner" "perfect_onehot_prob" "perfect_distributional_prob"; do
                        diversity_type="none"
                        for clustering_type in "none" "spectral"; do
                            name=$basename-sigma_$sigma-alpha_$alpha-confidence_type_$confidence_type-clustering_type_$clustering_type
                            bash launch_slurm_job_on_cpu.sh $name "python run.py --name $name --seed $seed --N $N --datatype $datatype --K $K --learner $learner --sampler $sampler --sigma $sigma --alpha $alpha --confidence_type $confidence_type --diversity_type $diversity_type --clustering_type $clustering_type"
                        done
                    done

                    confidence_type="learner"
                    diversity_type="optimal"
                    for clustering_type in "none" "spectral"; do
                        name=$basename-sigma_$sigma-alpha_$alpha-confidence_type_$confidence_type-clustering_type_$clustering_type-diversity_type_$diversity_type
                        bash launch_slurm_job_on_cpu.sh $name "python run.py --name $name --seed $seed --N $N --datatype $datatype --K $K --learner $learner --sampler $sampler --sigma $sigma --alpha $alpha --confidence_type $confidence_type --diversity_type $diversity_type --clustering_type $clustering_type"
                    done
                    
                done
            done
            
        done
    done
done