# Code

## Key Files and Directories

- **`Algorithms/`**: Contains the code for each algorithm candidate.

- **`Env/`**: Contains the code for building the dyadic simulation environment to test the algorithms.

- **`RunExp.py`**: Runs the experiment for a specific algorithm. An example usage is as follows:

    ```bash
    python Code/RunExp.py \
        --NoTrend \
        --save "results/reward_learning.csv" \
        --config "config_general_run.json" \
        --run_one "RewardLearningNewCare" \
        --seed 2 \
        --softmax_tem \
        --treat0 0.5 \
        --treat1 0.5 \
        --treat2 0.5 \
        --treat3 0.5 \
        --mediator 1
    ```

    - **`save`**: Specifies the location to save the resulting CSV files with the actions and rewards.
    - **`config`**: Points to a configuration file specifying the algorithms. This file should contain the algorithm specified in `run_one`.
    - The experiments for testing various algorithms under different environments are present in `Experiment_Test_Algs`, which contains repeated calls to the `Code/RunExp.py` file.

    See [`Env/envargs.py`](https://github.com/StatisticalReinforcementLearningLab/ADAPTS-HCT-AIME/blob/main/Code/Env/envargs.py) for definitions of the environment parameters (e.g., `treat0`, `seed`, etc.).

    **Note:** In the paper, we use a single hyperparameter, i.e., `treat0 = treat1 = treat2 = treat3`.

- **`compute_opt.py`**: Generates the optimal policy given a set of environment settings and an optimal policy candidate, and stores the resulting policy in `Opt_Policy/`.

    The experiments for testing various optimal policy approximation candidates in a range of environments are in `Experiment_Test_Opt_Policy/`. To run this file for a single environment and optimal policy approximation, use the following command:

    ```bash
    python Code/compute_opt.py \
        --treat0 1.20 \
        --treat1 1.20 \
        --treat2 1.00 \
        --treat3 1.20 \
        --bNoise 2.40 \
        --n 100 \
        --rep 1 \
        --horizon 14 \
        --optimal_policy singleagent \
        --mediator 0 \
        --single_agent_gamma 0
    ```

The environment parameters (`treat0`, `treat1`, `treat2`,`treat3`, `bNoise`, `n`, `rep`, `horizon`) are found in `Env/envargs.py` as specified earlier. `optimal_policy` specifies the candidate for the approximation which can be [`only_first_three_features`,`only_first_five_features`, `only_first_six_features`, `singleagent`]. The first three use offline fitted Q-learning on a dataset generated under random policy where the features are either the first three, five, or six features of the state. The `singleagent` approximation uses a flattened single agent with an 8-dimensional action space to approximate the optimal policy. The agent is trained using RLSVI and learns a linear approximation for the Q values.