# FYP Codebase
## [INFO] Release Note
❗ As previous codebase was pretty messy and hard to read, this codebase is the refactored version of previous one.
In this codebase, the training pipeline is neatly implemented and the code is modularized to support further experiments. The code used for previous 
experiments are obsolete and will not be included in this repo. For future release of the code base, I will write the configuration of each experiemnt to the config folder,
for the ease of reproducibility.
 


⏳ TODO:

* High Priority:
  Define features and disentanglement of features in a classifier via designing the optimization target for it and obtain its mask.

* Low Priority:
  * Add gradient to pruned part of network to this codebase DSD. 
    * for old experiments, currently  of no use
  


## Command Line Arguments

The specify training configuration and hyper-parameters. The configurations can be passed through config file under the
config folder, also the command line. The command line arguments will override the setup specified in the config file.
For example:

```shell
python main.py  --config config/config.yml --batch-size 32 --epochs 100
```

### General

* configs: The path to configuration file
* result-dir: Directory to store the training results
* exp-name: Name of the experiment, the sub directory will be created under result-dir
* exp-mode:  The mode for experiment, can be train, prune.
    * train: Train the model from scratch
    * prune: Prune the model
* print-freq: Frequency of printing training status in training epoch

### Model and Dataset

* arch: The model architecture
* pretrained: flag to use pre-trained model for resnet.
* num-classes: Number of classes
* dataset: The dataset to use
* data-dir: The root directory to store the all datasets. Each dataset is a sub-directory under data-dir 

### Hyper-parameters for training

* batch-size: Batch size for training
* test-batch-size: Batch size for testing
* epochs: Number of epochs to train
* warmup-epochs: Before using large learning rate to train the model, use small learning rate to warm up the training
  for several epochs.
* warmup-lr: The learning rate used for warm up epochs
* lr: Learning rate
* lr-schedule: Learning rate schedule
* optimizer: The optimizer to use, support "sgd", "rmsprop", "adam"

### Environment

* gpu: Indicate the id of gpu to use
* no-coda: Disable GPU
* seed: Random seed for result reproducibility

### Model Pruning

* prune-rate: The prune rate k for each layer, only for supported model (models/resnet.py), k percent of the
  weights are preserved.

