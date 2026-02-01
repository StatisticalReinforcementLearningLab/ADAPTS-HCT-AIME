# ADAPTS-HCT-AIME

This is the repository for the paper *Reinforcement Learning on AYA Dyads to Enhance Medication Adherence*. 

The structure of the repository is as follows:

1. `Code/` contains the code for the different candidate algorithms and for the dyadic environment (in subdirectories `Algorithms/` and `Env/` respectively).
2. `Experiment_Test_Algs/` contains the experiments to obtain and plot the cumulative rewards under different possible algorithms.
3. `Experiment_Tune_Ctreat/` contains the experiments to tune the hyperparameter $C_\text{Treat}$ which controls the treatment effects and is imputed to obtain the STEs of 0.15, 0.3, and 0.5.
4. `Experiment_Test_Opt_Policy/` contains the experiments to test the different optimal policy approximation candidates.
5. `Model_Fitting/` contains the coefficients fitted through GEE for the dyadic environment models.
6. `Opt_Policy/` contains the pickle files for the optimal policy approximation run under different environments.

**Each of the directories [`Code/`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-RL-Algorithm-Design/tree/submission/Code#code), [`Experiment_Test_Algs/`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME), [`Experiment_Tune_Ctreat/`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME/tree/main/Experiment_Tune_Ctreat), and [`Experiment_Test_Opt_Policy/`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME/tree/main/Experiment_Test_Opt_Policy) contains further detail on the code structure and running instructions.**

ROADMAP Dataset was used to fit the simulator models in the project. The dataset is available for download [here](https://deepblue.lib.umich.edu/data/concern/data_sets/ht24wk394?locale=en). 
