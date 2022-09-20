cmd=${1:-train-visualize}
echo $cmd 
data_dir=/home/azureuser/cloudfiles/data/datastore/pathtodata
output_dir=/home/azureuser/cloudfiles/code/Users/alias/pathtooutputdir

USER_TOP_K=100

train_cmd="python main.py \
    --input-path ${data_dir}/cf_trainset.csv \
    --test-path ${data_dir}/cf_valset.csv \
    --output-dir ${output_dir} \
    --save-npy \
    --save-user-doc-preds \
    --user-top-k ${USER_TOP_K}\
    --conv-metric recall_${USER_TOP_K} \
    --do-evaluation \
    --expand-user-coverage \
    --input-path-expandusercoverage ${data_dir}/cf_trainset_prism_extra_combined.csv \
    --test-path-uniqueextra  ${data_dir}/cf_testset_uniqueextra.csv \
    --test-path-uniquespecific  ${data_dir}/cf_testset_uniquespecific.csv \
    --doc-embeddings-visual-path  ${data_dir}/visual_embeddings.tsv \
    --doc-embeddings-text-path  ${data_dir}/text_embeddings.tsv \
    --report-diversity \
    --read-frequency \
    --save-log-to-file "

eval_cmd="${train_cmd}--evaluation-only"

visualize_cmd="cd ../util && 
                    python visualizer.py \
                            --visualize-prism \
                            --visualize-extra \
                            --visualize-specific \
                            --data-dir ${data_dir} \
                            --result-dir ${output_dir}"


if [[ $cmd = "train" ]]
then 
    echo "RUNNING CF TRAINING .... "

    echo ${train_cmd}
    eval ${train_cmd}
fi
if [[ $cmd = "train-visualize" ]]
then 
    echo "RUNNING TRAINING AND VISUALIZATION ---"
    
    echo ${train_cmd}
    eval ${train_cmd}

    echo ${visualize_cmd}
    eval ${visualize_cmd}
fi
if [[ $cmd = "evaluate-visualize" ]]
then 
    echo "RUNNING EVALUATION AND VISUALIZATION ----"
    echo ${eval_cmd}
    eval ${eval_cmd}
 
    echo ${visualize_cmd}
    eval ${visualize_cmd}
fi
if [[ $cmd = "evaluate" ]]
then 
    echo "RUNNING EVALUATION AND VISUALIZATION ----"
    echo ${eval_cmd}
    eval ${eval_cmd}
 
fi
if [[ $cmd = "visualize" ]]
then 
    echo "RUNNING EVISUALIZATION ----"

    echo ${visualize_cmd}
    eval ${visualize_cmd}
fi

