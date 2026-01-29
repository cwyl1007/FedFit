# FedFit: Federated Dynamic Sparse Training via Fisher Information scoring

This repository provides an implementation for paper "FedFit: Federated Dynamic Sparse Training via Fisher Information scoring".


## Setup the Environment
```python
conda create -n fedpruning python=3.10
conda activate fedpruning
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda mpi4py
pip install -r requirements.txt
```

## Download the Data
Navigate to the desired dataset's folder in */data* 
```python
cd data/cifar10
sh download_cifar10.sh
cd ../..
```

## GPU Mapping
In *gpu_mapping.yaml* inside *experiments/distributed/fedfit*, modify *four_gpu* to have the correct number of GPUs and correct number of processes.  
e.g For four GPUs and 11 processes (10 clients + 1 server, where clients is client_num_per_round)
```python
four_gpu: [2, 3, 3, 3]
```

## Running the Code
To run the code, navigate to the *experiments/distributed/fedfit* and run the below command 
```python
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedfit_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [target_density] [initial_lr]  {--delta_T , --T_end, --partition_alpha, --num_eval, --frequency_of_the_test}
```
Where **[param]** marks required parameters, and **{params}** are optional parameters and,  
```python
[gpus] specifies which GPUs to use.
[model] is the name of the model.
[dataset] is the name of the dataset.
[client_num_in_total] is the total number of clients.
[client_num_per_round] is the number of clients selected per round.
[comm_round] is the number of communication rounds.
[epochs] is the number of local epochs.
[target_density] is the target density for the sparse model.
[initial_lr] is the initial learning rate.
{--delta_T} is the interval rounds between two adjustment round.
{--T_end} is the end round number for adjustment round.
{--partition_alpha} refers to the partition alpha, higher partition alpha makes lower degree of data heterogeneity
{--num_eval}is the number data samples for validation, -1 means using the whole testing dataset.
{--frequency_of_the_test} the frequency to test/validate the performance during the training, using num_eval data samples
```

For example, a quick run of FedFit on resnet18 backbone with cifar10, on 100 clients
```python
cd experiments/distributed/fedfit
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedfit_distributed_pytorch.sh resnet18 cifar10 100 10 500 1 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 
```
