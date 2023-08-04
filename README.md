# Mitigating Label Biases for In-context Learning

This is a codebase of our [ACL 2023 paper](https://arxiv.org/abs/2305.19148). It is developed based on the [codebase](https://github.com/tonyzhaozh/few-shot-learning/tree/main) by Tony Z. Zhao to perform few-shot in-context learning on text classification tasks using large language models such as [GPT-3](https://arxiv.org/abs/2005.14165).

You can run this codebase with GPT-3, GPT-J, GPT-2, and potentially any other language model available in [HuggingFace Transformers](https://huggingface.co/models). To run the codebase with GPT-3, you should place your API key into a file named `openai_key.txt`.

Running this codebase will report results with [domain-context calibration](https://arxiv.org/abs/2305.19148) (our proposed method), [contextual calibration](http://arxiv.org/abs/2102.09690), and without any calibration method.

## Dependencies

This code is written using PyTorch and [HuggingFace's Transformer repo](https://github.com/huggingface/pytorch-transformers). Running the code locally requires GPUs (except for OpenAI models like GPT-3), but with some minor modification it is possible to adapt the code to devices with only CPUs.

## Installation

The easiest way to install the code is to create a fresh anaconda environment:
```bash
conda create -n labelbias python=3.9
conda activate labelbias
pip install -r requirements.txt
```

## Data
All the datasets we used are publicly available text classification datasets. Running the code will automatically download datasets from Huggingface except for ones already exists in the `data` folder.

## Reproducing Our Results

Here is how to replicate the results from our paper for GPT-3. To replicate the results on SST-2 (8-shots, 5 different in-context examples, subsample 500 samples from the original test set):
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py \
--model="davinci" \
--dataset="sst2" \
--num_seeds=5 \
--all_shots="8" \
--api_num_log_prob 100 \
--subsample_test_set 500 \
--recompute_probs
```

To test on the whole test set, remove `--subsample_test_set 500`.

## Overview of Codebase

### Data
The `data` folder contains the raw data for a part of supported tasks (other tasks are handled by the Huggingface dataset library). If you'd like to add your own task, you can either add the data into that folder or use datasets from [Huggingface](https://huggingface.co/datasets). The code for loading a dataset (from the `data` folder or from Huggingface), as well as defining the prompt format for a task, is in `data_utils.py`. You can refer to the loader functions for existing tasks for writing one for a new task.

### Run Scripts
The run scripts, e.g., `run_classification.py`, contain the code for randomly sampling the examples to use in the prompt, calling the models, the necessary evaluation metrics, and more. Inside the run script, you can set the parameters for the experiments using the command line arguments.

For all experiments, we save and pickle the outputs of the model (including the raw predictions of the model, the evaluation metrics, e.g., accuracy, estimated priors, etc). This makes doing a post-hoc analysis very fast.


## References

Please consider citing our work if you found this code or our paper beneficial to your research.
```
@article{Fei2023MitigatingLB,
  title={Mitigating Label Biases for In-context Learning},
  author={Yu Fei and Yifan Hou and Zeming Chen and Antoine Bosselut},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.19148},
  url={https://api.semanticscholar.org/CorpusID:258967265}
}  	
```