# Experiment_Test_Algs

This directory contains the code for running experiments to obtain and plot the cumulative rewards under different algorithms.

## How to Specify and Run Experiments

**To reproduce our results (Figure 2 from the paper), follow steps 2-4. To specify additional experiments, you may start with step 1.**

1. Create a new file within `configs/design_decisions/` specifying your experiment setup and algorithms to run. See the example `configs/design_decisions/Run_All_STE.json` below:

    ```json
    {
        "env_list": ["Mediator0_03", "Mediator0_05", "Mediator0_015", 
                     "Mediator1_03", "Mediator1_05", "Mediator1_015", 
                     "Mediator2_03", "Mediator2_05", "Mediator2_015",
                     "Mediator-1_03", "Mediator-1_05", "Mediator-1_015"],
        "n": 25,
        "rep": 1000,
        "algorithm_list": ["MRT", "SingleAgent_RwdNaive_Pool_Sigm0.5_Lambda0.75", "RewardLearning", "RLSVI_Inf_Gamma_0.5", "RewardLearningNewCare"]
    }
    ```

   The definitions of each key in this JSON file are as follows:
   - **`env_list`**: The list of environments in which you want to test your algorithms. Each element in this list must be further defined in another JSON file named `configs/environment_template{optional suffix}.json`.
   - **`n`**: The number of dyads in the simulated trial.
   - **`rep`**: The number of repetitions for which to run the trial.
   - **`algorithm_list`**: The list of algorithms to test in each of the environments specified in `env_list`. Each element of `algorithm_list` must be further defined in another JSON file named `configs/shared_algorithms{_optional suffix}.json`.

   The definitions of parameters in `shared_algorithms{_optional suffix}.json` can be found in [`Code/Algorithms/algargs.py`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME/blob/main/Code/Algorithms/algargs.py), and the definitions of parameters in `configs/environment_template{optional suffix}.json` are specified in [`Code/Env/envargs.py`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME/blob/main/Code/Env/envargs.py).

2. Ensure you are in the root of the repository (i.e., `cd ADAPTS-HCT-RL-Algorithm-Design`), and run:
    ```bash
    python Experiment_Test_Algs/scripts/generate_experiment_configs_env.py
    python Experiment_Test_Algs/scripts/generate_experiment_configs.py
    ```
   These commands will create the directories for the specified experiments and place the relevant configuration files within those directories. These are stored in `Experiment_Test_Algs/experiments`.

3. Run:
    ```bash
    sh Experiment_Test_Algs/scripts/submit-jobs.sh
    ```
   This script will check for all configuration files within the experiments directory (i.e., all `Experiment_Test_Algs/experiments/*/config_general_run.json` files) and submit SLURM jobs for experiments that have not already been run.

4. Finally, create plots of the cumulative outcomes for each component (AYA, care partner, and relationship) by running:
    ```bash
    python Experiment_Test_Algs/scripts/visualization.py
    ```
   This will generate the plots `All_Distress.png`, `All_Relationship.png`, and `All_Rewards.png` within each experiment's subdirectory in `experiments/`.
   
   The plots used for Figure 2 in the paper are `Experiment_Test_Algs/experiments/Run_All_STE/Mediator0_05/All_Rewards.png`, `Experiment_Test_Algs/experiments/Run_All_STE/Mediator0_15/All_Rewards.png`, and `Experiment_Test_Algs/experiments/Run_All_STE/Mediator0_03/All_Rewards.png`. 

---

## Notes

1. The CSV files containing the rewards and actions for all experiments will be saved in the global directory `Experiment_Test_Algs/results/` with filenames indicating the experiment setup and algorithm run (e.g., `Experiment_Test_Algs/results/save_t1.5_1.5_1.5_1.5_s1_soft100_n25_rep1000_ME2_MRT.csv`).
2. Each experiment subdirectory will contain an `experiments_run.json` file, which tracks reused results and newly generated results to avoid redundant runs.

---

## Key Files and Directories

- **`configs/design_decisions/`**: Contains JSON files for experiment-specific configurations.
- **`configs/environment_template*.json`**: Shared definitions of environments and their parameters.
- **`configs/shared_algorithms*.json`**: Shared definitions of algorithms and their parameters.
- **`scripts/generate_experiment_configs.py`**: Generates experiment directories and configurations.
- **`scripts/submit_jobs.sh`**: Submits jobs to SLURM and tracks existing results to avoid redundant runs.
- **`scripts/submit_local.sh`**: A local version of `scripts/submit_jobs.sh` that runs each algorithm sequentially.
- **`results/`**: Global directory where all results are saved.
- **`visualization.py`**: Visualizes the cumulative outcomes for each component in each experiment.
- **`del_old_run.py`**: Deletes existing results from `results/` for a given experiment configuration.
