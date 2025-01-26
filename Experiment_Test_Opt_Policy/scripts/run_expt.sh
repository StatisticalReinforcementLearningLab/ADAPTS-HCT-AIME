#!/bin/bash
#SBATCH --mail-type=FAIL

alg=$1
SCRIPT_DIR=$2

GLOBAL_RESULTS_DIR="Experiment_Test_Opt_Policy/results"
mkdir -p "$GLOBAL_RESULTS_DIR"

config_file="$SCRIPT_DIR/config_general_run.json"
seed=$(jq -r '.seed' "$config_file")
softmax_tem=$(jq -r '.softmax_tem' "$config_file")
treatments=($(jq -r '.treatments[]' "$config_file"))
mediator=$(jq -r '.mediator' "$config_file")
n=$(jq -r '.n' "$config_file")
rep=$(jq -r '.rep' "$config_file")

treat0=${treatments[0]}
treat1=${treatments[1]}
treat2=${treatments[2]}
treat3=${treatments[3]}

filename="save_t${treat0}_${treat1}_${treat2}_${treat3}_s${seed}_soft${softmax_tem}_n${n}_rep${rep}_ME${mediator}.txt"

echo "Running $alg with seed=$seed, softmax_tem=$softmax_tem, treatments=[$treat0, $treat1, $treat2, $treat3], mediator=$mediator"
echo saving to $GLOBAL_RESULTS_DIR/$filename

python Code/RunExp.py \
    --NoTrend \
    --sequential \
    --save "$GLOBAL_RESULTS_DIR/$filename" \
    --config "$config_file" \
    --run_one "$alg" \
    --seed $seed \
    --softmax_tem $softmax_tem \
    --treat0 $treat0 \
    --treat1 $treat1 \
    --treat2 $treat2 \
    --treat3 $treat3 \
    --mediator $mediator 

