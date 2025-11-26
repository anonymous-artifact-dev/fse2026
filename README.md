# Sentiment Analysis in Software Engineering: How Far Are We with Data Augmentation

An implementation of the paper "Sentiment Analysis in Software Engineering: How Far Are We with Data Augmentation" submitted to the FSE 2026.

We release our artifacts of Data Augmentation (Generation) and Fine-tuning to facilitate further resarch and adoption.

## Requirements
We have two environments, fine-tuning SLMs and LLMs with QLoRA.

### SLMs
```
pip install -r requirements.txt
```

### LLMs

```
conda create -n qlora python=3.10.18
conda activate qlora
conda install -r requiremeㅜts_qlora.txt
```

-----------

## Data Augmentation Techniques
We implemented each data augmentation approaches according to their open source codes in GitHub repositories.

These are studies we have studied.
- SSMBA (Self-Supervised Manifold Based Augmentation) - [SSMBA: Self-Supervised Manifold Based Data Augmentation for Improving Out-of-Domain Robustness](https://aclanthology.org/2020.emnlp-main.97/)
- AEDA (An Easier Data Augmentation) - [AEDA: An Easier Data Augmentation Technique for Text Classification](https://aclanthology.org/2021.findings-emnlp.234/)
- C2L (Causally Contrastive Learning) - [C2L: Causally Contrastive Learning for Robust Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/21296)
- TextSmoothing - [Text Smoothing: Enhance Various Data Augmentation Methods on Text Classification Tasks](https://aclanthology.org/2022.acl-short.97/)


## Evaluation

```
bash run.sh
```

The ```run.sh``` is a script for all experiment including augmenation, fine-tuning, prompting, and evaluation.
```linux
# run.sh

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

```

The augmented sets by each data augmentation are saved inside ```\temp``` folder with format, ```{dataset}_{da}.csv``` that you can check how training set is augmented.<br />
Each checkpoints of fine-tuned models are also saved inside ```\temp``` folder with format, ```train_{dataset}_{da}_{model}```.<br />
Results of prompting are saved insied ```\result``` folder by same format, ```chatgpt_{dataset}_{da}_{k}_{model}.txt```<br />

### If you want to evaluate with a specific case, follow the script below.<br />
```
# SLM
CUDA_VISIBLE_DEVICES=0,1,2 python train.py -dataset "dataset" -da "da" -model "model"

# LLM Fine-tuning
CUDA_VISIBLE_DEVICES=0,1,2 python llm.py -dataset "dataset" -da "da" -model "model"

# LLM Prompting
CUDA_VISIBLE_DEVICES=0,1,2 python llm.py -dataset "dataset" -da "da" -model "model"
```
For arguments, you can put as below.<br />

dataset: ```app```, ```so```, ```github```, ```jira```, ```gerrit```, ```tweets```, ```tweets_p```, and ```tweets_n```<br />
da: ```none```, ```ssmba```, ```aeda```, ```c2l``` and ```ts```<br />
model: <br />
```deberta```, ```xlnet```, and ```t5``` for SLM<br />
```codegen```, ```phi```, and ```deepseek``` for LLM Fine-tuning<br />
```gpt-4.1-nano``` and ```gpt-5-nano```<br />


