#!/bin/bash
# SLURM parameters
PART=murphy,shared             # Partition names
MEMO=40960                     # Memory required (40GB)
TIME=48:00:00                  # Time required (48 hours)

module load python/3.10.9-fasrc01

GLOBAL_RESULTS_DIR="Experiment_Test_Algs/results"
mkdir -p "$GLOBAL_RESULTS_DIR"

# FORCE_RUN=${1:-false}  # Argument to allow force run, default is false
FORCE_RUN=true
subdir=${1:-""}

# Array to store submitted job IDs
submitted_jobs=()

RUN_SCRIPT="Experiment_Test_Algs/scripts/run_expt.sh"

ORDP="sbatch --mem=$MEMO -n 1 -p $PART --time=$TIME"

#Iterate through all config_general_run.json files
find Experiment_Test_Algs/experiments$subdir -name "config_general_run.json" | while read config_file; do
    SCRIPT_DIR=$(dirname "$config_file")
    LOG_FILE="$SCRIPT_DIR/experiments_run.json"  #File to track reused and run results

    echo '{"reused": [], "new": []}' > "$LOG_FILE"

    algs=$(jq -r '.algs | keys | join(" ")' "$config_file")
    echo "----------------$config_file--------------------"
    # echo "Algorithms list: $algs"

    #Extract experiment parameters from the config file
    seed=$(jq -r '.seed' "$config_file")
    softmax_tem=$(jq -r '.softmax_tem' "$config_file")
    treatments=($(jq -r '.treatments[]' "$config_file"))
    mediator=$(jq -r '.mediator' "$config_file")
    treat0=${treatments[0]}
    treat1=${treatments[1]}
    treat2=${treatments[2]}
    treat3=${treatments[3]}

    n=$(jq -r '.n' "$config_file")
    rep=$(jq -r '.rep' "$config_file")

    for alg in $algs; do
        # if [ "$mediator" = true ]; then
        filename="save_t${treat0}_${treat1}_${treat2}_${treat3}_s${seed}_soft${softmax_tem}_n${n}_rep${rep}_ME${mediator}_${alg}.csv"
        if [ "$FORCE_RUN" = false ]; then
            #Skip if the result already exists
            if [ -f "$GLOBAL_RESULTS_DIR/$filename" ]; then
                # echo "Results for $alg already exist. Skipping..."
                jq --arg file "$filename" '.reused += [$file]' "$LOG_FILE" > tmp.json && mv tmp.json "$LOG_FILE"
                continue
            fi

            # Check if the job is already submitted
            if [[ " ${submitted_jobs[@]} " =~ " $filename " ]]; then
                # echo "Job for $filename is already submitted. Skipping..."
                jq --arg file "$filename" '.reused += [$file]' "$LOG_FILE" > tmp.json && mv tmp.json "$LOG_FILE"
                continue
            fi
        fi

        # echo "Submitting job for algorithm: $alg"
        
        JOBN="AlgRun_${alg}"
        OUTF=$SCRIPT_DIR/"$JOBN.out"
        ERRF=$SCRIPT_DIR/"$JOBN.err"

        
        echo "Generating: $filename"
        bash $RUN_SCRIPT $alg $SCRIPT_DIR

        # Add the job ID to the submitted_jobs array
        submitted_jobs+=("$filename")

        jq --arg file "$filename" '.new += [$file]' "$LOG_FILE" > tmp.json && mv tmp.json "$LOG_FILE"
    done
done

