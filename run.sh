#!/bin/bash

echo "Script for SLM is Running"

datasets=("app" "so" "github" "jira" "gerrit" "tweets" " tweets_n" "tweets_p")
da=("none" "aeda" "ssmba" "c2l" "ts")
models=("deberta" "xlnet" "t5")

for dataset in "${datasets[@]}"; do
	for da in "${da[@]}"; do
		for model in "${models[@]}"; do
			echo "Running SLM with dataset=$dataset, da=$da, model=$model"
				CUDA_VISIBLE_DEVICES=0,1,2 python train.py -dataset "$dataset" -da "$da" -model "$model"
		done
	done
done

echo "Script for LLM-FT is Running"

datasets=("app" "so" "github" "jira" "gerrit" "tweets" " tweets_n" "tweets_p")
da=("none" "aeda" "ssmba" "c2l" "ts")
models=("codegen" "phi" "deepseek")

for dataset in "${datasets[@]}"; do
        for da in "${da[@]}"; do
                for model in "${models[@]}"; do
                        echo "Running LLM with dataset=$dataset, da=$da, model=$model"
                                CUDA_VISIBLE_DEVICES=0,1,2 python llm.py -dataset "$dataset" -da "$da" -model "$model"
                done
        done
done


echo "Script for LLM Prompting is Running"

export OPENAI_API_KEY="YOUR_API_KEY"

DATASET_DIR="dataset"
RESULTS_DIR="results"
models=("gpt-4.1-nano" "gpt-5-nano")
damethods=("c2l" "aeda" "ssmba" "ts" "c2l")
fewshot_ks=(0 1 2 4)

for model in "${models[@]}"; do
    for da in "${damethods[@]}"; do
        for k in "${fewshot_ks[@]}"; do
            echo "Running Prompt with model=$model, DA=$da, K=$k"
            PYTHONNOUSERSITE=1 python prompt.py \
                --dataset_dir "$DATASET_DIR" \
                --results_dir "$RESULTS_DIR" \
                --model "$model" \
                --da "$da" \
                --fewshot_k "$k"
        done
    done
done
