# FedFit: A Fisher Information Based Federated Pruning Method

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
```
cd data/cifar10
sh download_cifar10.sh
cd ../..
```

## GPU Mapping
In *gpu_mapping.yaml* inside *experiments/distributed/fedfit*, modify *four_gpu* to have the correct number of GPUs and correct number of processes
e.g For four GPUs and 11 processes (10 clients + 1 server, where clients is client_num_per_round)
```
four_gpu: [2, 3, 3, 3]
```

## Running the Code
To run the code, navigate to the *experiments/distributed/fedfit* and run the below command 
```
CUDA_VISIBLE_DEVICES=[gpus] sh run_fedfit_distributed_pytorch.sh [model] [dataset] [client_num_in_total] [client_num_per_round]  [comm_round] [epochs] [target_density] [initial_lr]  {--delta_T , --T_end, --partition_alpha, --num_eval, --frequency_of_the_test}
```
Where [] marks required parameters, and inside {} are optional parameters.

For example, a quick run of FedFit on resnet18 backbone with cifar10, on 100 clients
```
cd experiments/distributed/fedfit
CUDA_VISIBLE_DEVICES=0,1,2,3 sh run_fedfit_distributed_pytorch.sh resnet18 cifar10 100 10 500 1 0.1 0.1 --delta_T 10 --T_end 300 --num_eval 128 --frequency_of_the_test 10 
```
