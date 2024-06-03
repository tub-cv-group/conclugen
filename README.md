Multi-Task Multi-Modal Self-Supervised Learning for Facial Expression
Recognition - Published CVPR 2024 workshop ABAW
==============================

_Marah Halawa*, Florian Blume*, Pia Bideau, Martin Maier, Rasha Abdel Rahman, Olaf Hellwich_
*equal contribution

---

You can access the preprint [here](https://arxiv.org/abs/2404.10904)

![overview_v3(1)_page-0001](https://github.com/tub-cv-group/conclugen/assets/2170192/2d82befe-f01e-4684-8f91-78c65c8e96eb)

Human communication is multi-modal; e.g., face-to-face interaction involves auditory signals (speech) and visual signals (face movements and hand gestures). Hence, it is essential to exploit multiple modalities when designing machine learning-based facial expression recognition systems. In addition, given the ever-growing quantities of video data that capture human facial expressions, such systems should utilize raw unlabeled videos without requiring expensive annotations. Therefore, in this work, we employ a multitask multi-modal self-supervised learning method for facial expression recognition from in-the-wild video data. Our model combines three self-supervised objective functions: First, a multi-modal contrastive loss, that pulls diverse data modalities of the same video together in the representation space. Second, a multi-modal clustering loss that preserves the semantic structure of input data in the representation space. Finally, a multi-modal data reconstruction loss. We conduct a comprehensive study on this multimodal multi-task self-supervised learning method on three facial expression recognition benchmarks. To that end, we examine the performance of learning through different combinations of self-supervised tasks on the facial expression recognition downstream task. Our model ConCluGen outperforms several multi-modal self-supervised and fully supervised baselines on the CMU-MOSEI dataset. Our results generally show that multi-modal self-supervision tasks offer large performance gains for challenging tasks such as facial expression recognition, while also reducing the amount of manual annotations required. We release our pre-trained models as well as source code publicly.

---
# Quickstart

1. Clone the repository

2. Assuming you have `docker` installed, you can execute from the project's directory the script:

    `. ./bash/run.sh model_name dataset_name`

    where `model_name` can be one of the following:

    - `conclugen` (model with all three losses)
    - `contrast_cluster` (combination of multi-modal contrastive and clustering)
    - `contrast_generative` (combination of multi-modal contrastive generative)
    - `contrast_only` (multi-modal contrastive with all three modalities)
    - `contrast_only_video_audio` (multi-modal contrastive with video-audio)
    - `contrast_only_video_text` (multi-modal contrastive with video-text)
    - `generative_only` (generative loss only with three modalities)
    - `simclr` (video version of SimCLR)

    and `dataset_name` can be one of the following:

    - `caer`
    - `meld`
    - `mosei`

    The code will automatically download the model weights and datasets (precomputed features of the backbones) and run testing for FER on the respective dataset.

Extra:

3. If you want to log to CometML, also set `COMET_API_KEY` to your CometML API key and `COMET_WORKSPACE` to your CometML workspace. In this case, you also need to uncomment the lines `- class_path: loggers.CometLogger` in `configs/conclu.yaml` and `configs/simclr.yaml`

    The remaining environment variables will be created by the script.

---
# Table of Contents

1. [Introduction](#introduction)

2. [Structure of Project](#structure-of-project)

3. [Environments](#environments)

4. [Running the Code](#running-the-code)

5. [Configuration](#configuration)

6. [Implementation](#implementation)

7. [Logging](#logging)

8. [Training](#training)

9. [Testing & Evaluation](#testing--evaluation)

10. [Docker](#docker)

11. [Slurm](#slurm) 

12. [Issues, Common Pitfalls & Debugging](#issues-common-pitfalls--debugging)

13. [Developtment Advice](#development-advice)

14. [Notes](#notes)

---
# Introduction

The code is structured along the following frameworks and tools:

### [Pytorch](www.pytorch.org)

A machine learning library for implementing artificial neural networks.

### [Pytorch Lightning](www.pytorchlightning.ai)

Simply put what Keras is for Tensorflow - **Automation and structuring of Pytorch training, testing, evaluation, etc.** Usually, every researcher has their own way of implementing a Pytorch training loop, inference, etc. Pytorch Lightning helps to structure this in a unified way and provides extensive logging functionality. This project also draws on the [Pytorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html) which makes running Pytorch code through config files a lot easier. Please read the documentation of the CLI first. You don't have to understand it completely but just how it works in general.

### [jsonargparse](https://jsonargparse.readthedocs.io/en/stable/index.html)

jsonargparse is included in Pytorch Lightning's CLI. The general idea is that the configuration on the command line is based on the init functions of classes or function arguments. This way, it is entirely clear which arguments can be provided and no guessing is necessary. jsonargparse can also take care of instantiating classes. Arguments can be provided directly in the command in the terminal or by pointing to a yaml file.

### [MLflow](www.mlflow.org)

MLflow is a machine learning life-cycle managment platform. Essentially, it provides local and remote logging functionality, a nice UI to explore your experiments and a way of running experiments from command line using pre-defined entrypoints (you'll see about this later). Running the project through MLflow will automatically create any needed Anaconda environment or run it within the proper Docker or Singularity container. More specifically, you'll need [my branch of mlflow](https://github.com/florianblume/mlflow/releases/download/v1.4.1/mlflow-1.4.1-py3-none-any.whl).

### [CometML](www.comet.ml)

CometML is similar to MLflow but since it's a paid service it provides far more features and is more usable. The downside is that you have less control over your data. Accounts for academics are free. We use a combination of MLflow and CometML logging to have the convenicen of online backups and better evaluation tools but also have local files and data.

Both log the full code together will model weights, configs, losses, metrics and everything else you additionally manually log. This way, nothing gets lost. No inspection of text files where you wrote the final classification accuracy to.

The next steps will guide you in how to setup the environmnents and run the code.


---
# Structure of Project

```
├── bash                    -- some useful bash scripts
├── configs                 -- the yaml configs
│   ├── eval                -- yaml configs for evaluation
│   ├── model               -- yaml configs for models
│   └── slurm               -- yaml configs for slurm
├── docker                  -- Dockerfiles
├── docs                    -- mostly empty, shame on me
├── experiments             -- space to put your experiments
├── models                  -- checkpoints of trained models
├── notes                   -- also mostly empty
├── singularity             -- folder to convert the Docker image in
└── src                     -- source folder
    ├── callbacks           -- callbacks which can be used on the trainer or model
    ├── data                -- all data related implementations
    │   ├── datamodules     -- all datamodules
    │   ├── datasets        -- all datasets
    │   └── transforms      -- transforms for data
    ├── evaluation          -- evaluation scripts
    ├── loggers             -- the supported implemeneted loggers
    ├── losses              -- custom losses
    ├── models              -- all model files
    │   └── backbones       -- model backbones
    ├── prototyping         -- space to perform some prototyping
    ├── utils               -- all sorts of utility functions
    └── visualization       -- all visualization related stuff
 ```

---
# Environments

*Generally, it is advised to simply use MLflow to run the code since it will take care of setting up any necessary environment. To use MLflow for running the code, please be sure to install [my branch of mlflow](https://github.com/florianblume/mlflow/releases/download/v1.4/mlflow-1.4-py3-none-any.whl):*

    pip install https://github.com/florianblume/mlflow/releases/download/v1.4.1/mlflow-1.4.1-py3-none-any.whl

You have three options to use as environment:

## 1. [Anaconda](https://www.anaconda.com/)

To setup the Anaconda environment simply do

    conda env create --name [name your env] -f [environment file]

where you replace `[environment file]` with one of the files available in the project root.

## 2. [Docker](www.docker.io)

You can also use the Docker images found at [Docker hub](https://hub.docker.com/repository/docker/florianblume/facial-expression-recognition) or use the Dockerfiles in the `docker` directory to build it yourself. There's also a script in the `bash` directory which contains the command to build the Docker image (be careful, it is configured to delete the cache so it will always build the Dockerfile). The command to build it is

    docker build -t florianblume/facial-expression-recognition:cuda11.3-pytorch1.11 -f docker/cuda11.3-pytorch1.11/Dockerfile --no-cache .

When you want to use Docker you might have to create specific folders before running the code and mount them manually/through the entrypoints file since Docker has no direct access to the underlying filesystem. I.e., create the folders on your local machine and modify access rights like so

    chmod 766 /path/to/the/folder/to/mount

## 3. [Singularity](https://sylabs.io/)

To use Singularity, you can build the Dockerfile and convert the resulting image to a Singularity image. This is especially useful if there is no access to Docker on your cluster. There are multiple guides online how to do the conversion.

---
# Running the Code

There are two main ways how to run the code: Through MLflow and through Python directly. They have a high overlap and essentially are pretty similar. The difference is, when you run the code through MLflow, you know exactly which parameters are allowed and MLflow will take care of either setting up the Anaconda environment or constructing and executing a proper Docke command which automatically mounts all necessary folders, attaches necessary environment variables and so on. It's therefore easier to run the code through MLflow.

For both ways, the central entrypoint of the project is `cli.py` in the `src` folder.

## 1. Through MLflow (suggested way)

To install MLflow, run

    pip install https://github.com/florianblume/mlflow/releases/download/v1.4.1/mlflow-1.4.1-py3-none-any.whl

### Running an Experiment

For MLflow, entrypoints are defined how to run the code. You can check them out in `MLproject_entrypoints.yaml`. Each key in the yaml dictionary is an entrypoint you can run which also defines the respective parameters. You can run the code like so

    mlflow run MLproject_conda -e [entrypoint] -P model=configs/model/[model].yaml -P data=configs/data/[data].yaml -P config=configs/to_edit.yaml

where you replace `[model]` by the model config's name (found in `configs/model` directory), `[data]` by the data config's name (found in `configs/data` directory). The `-P`s stem from the concept that you hard-define arguments for the entrypoints which removes uncertainty on the user-side what to provide. For `[entrypoint]` checkout `MLproject_entrypoints.yaml` but typically you would use

* `main`: for training and testing
* `slurm`: to run the code through slurm
* `inspect_batch_inputs`: to inspect the batch inputs
* ...

MLflow needs to take care of some argument conversions though, which is done in `mlflow_entrypoints.py`. This means, if you need to add an entrypoint (like I had to e.g. for automatic experiment downloading), you need to add it in `MLproject_entrypoints.yaml` and also in `mlflow_entrypoints.py`.

### Selecting an Environment

You can select different environments for MLflow (or create your own):

* `MLproject_conda` for running using Anaconda
* `MLproject_docker` for running using Docker
* `MLproject_slurm` so that you can define a different environment when running through slurm - this does not yet run the code using Slurm (checkout the Slurm part of the README for details)

MLflow will automatically take care of constructing an Anaconda environment if needed (hashes the environment.yaml file to see if something changed), pulls the respective Docker image, etc.

### Further Information

Checkout the [MLflow projects documentation](https://www.mlflow.org/docs/latest/projects.html) for more details on running projects through MLflow.


### Setting Up an Alias

```
alias mlmain='mlflow run MLproject_conda -e main
alias mlslurm='mlflow run MLproject_conda -e slurm
```

(or whatever combinations you need) to `.bashrc` you can run it like

    mlmain -P model=configs/model/[model].yaml -P data=configs/data/[data].yaml -P config=configs/to_edit.yaml

## 2. Through Python directly

MLflow essentially uses `cli.py` which you can also call directly:

    python src/cli.py --model=configs/model/[model].yaml --data=configs/data/[data].yaml --config=configs/to_edit.yaml

Note how the `-P`s were removed and instead the arguments are provided directly. Before you can run this command, make sure you activated your Anaconda environment (MLflow does this for you) or execute it within a Docker image.

If you provide an "entrypoint" after `python src/cli.py`, it will be executed here, too, instead of the main entrypoint. This means you can do

    python src/cli.py inspect_batch_inputs --model ...

Leaving the entrypoint out is equivalent to calling

    python src/cli.py main --model ...

---

## Under the Hood

What happens when you run the code like this is that

1. The CLI parses the arguments.
2. The CLI constructs the model and data class.
3. The CLI also constructs the [trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) class - the class that takes care of executing a training/validation/testing... loop.
4. The CLI executes the sequence of trainer commands (i.e. functions callable on the trainer) provided through the configuration. The trainer uses the model and datamodule to perform commands.

If you want to see what happends in detail, check out the `src/cli.py` file.

---
# Configuration

## Overview

**Some of the following text explains things that are only possible by running the code directly through Python and not through MLflow. They have the same capabilities though and it is noted where this applies.**

The project's configuration scheme is based on jsonargparse which is included by Pytorch Lightning's CLI. jsonargparse is built on top of Python's native argparse and extends its functionality. The general idea is that classes' init functions as well as function signatures define what can be configured from command line by directly providing the arguments or providing yaml files. For that to work, you always need to provide type annotations on the init functions of your classes.

[The Pytorch Lightning CLI documentation](https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html) is pretty good and deep. Look it up to get further details on how configuration works.

---

In the following, we'll explain firt how configuration works in Pytorch Lightning CLI and then move to how to use and edit the config files.

## The Basics

Pytorch Lightning CLI defines two default configuration keys: `model` and `data`. Additionally, you can provide **as many** `config` arguments as you want. For example, if your model class looks like this

```python
from models import AbstractModel

class MyModel(AbstractModel):

    __init__(self, batch_size: int, img_size: int = 128, **kwargs):
        # Important! We need to pass on the args to the superclass
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.img_size = img_size
```

then the keys configurable from the command line would look like the following:

    python src/cli.py --model.class_path models.MyModel --model.init_args.batch_size 32 --model.init_args.img_size 256

while you wouldn't need to define `img_size` as it's got a default value. You could also define these values in a config like so

```yaml
class_path: models.MyModel
init_args:
    batch_size: 32
```

and pass it like

    python src/cli.py --model model_config.yaml

Note that `--model` inserts the content of the config into the key `model` in the resulting config. If you pass the config file like

    python src/cli.py --config model_config.yaml

you need to change the config to

```yaml
model:
    class_path: models.MyModel
    init_args:
        batch_size: 32
```

This is important to know if you want to overwrite values with some config later in the command. For examlpe, in the command

    python src/cli.py --model model_config.yaml --config to_edit.yaml

`to_edit.yaml` would overwrite values *which are already defined in `model_config.yaml`*. `model_config.yaml` would, for examlpe, be the config from above without the `model` key, and `to_edit.yaml` could be

```yaml
model:
    init_args:
        batch_size: 64
```

Here you don't need the `class_path` entry anymore since it's provided already in `model_config.yaml` but you need the `model` key, as the config is passed with `--config` and not `--model`. This way, you can cascade configs and have a config you continuously edit to overwrite certain values without having to edit the default configs.

---

**Note 1:** You can pass `--config` multiple times, they will be loaded in sequence and later, i.e.

    python src/cli.py --config config1 --config config2 --config config3

will load `config1` first, then `config2`, and so on. The later configs overwrite values that are defined in both configs.

`--model` and `--data` can only be passd once though.

---

**Note 2:** The trainer can also be configured using `--trainer` but you will need this rarely.

---

### Notes on MLflow

MLflow doesn't give you the same flexibility when it comes to configuration. To ensure that entrypoints are well documented and allow to be run in one specific way which reduces uncertainty how to run the code, all possible arguments need to be defined in the `MLproject_entryopints.yaml` file. It's obvious that you cannot put all possible arguments in there (e.g. `model.init_args.batch_size` and so on). Instead, it only accepts the whole config files.

There is one option to pass such a config string through `-P strconfig=model.init_args.batch_size=32;model.init_args.img_size=128` and so on. But it is recommended for simplicity to make changes e.g. in a `to_edit.yaml` config.

## Model and Data Configuration

Any model you put in `src/models`, let it inherit from `AbstractModel` and put it into the `__init__.py` of the `models` module, will be available to use in the `class_path` part of the model config and to configure from the command line/in config files. Again, the configurable arguments are the typed arguments of the init function of the class.

Same holds for the data class. Put your `DataModule` in `src/data/datamodules`, let it inhert from `AbstractDataModule` (import via `from data.datamodules import AbstractDataModule`) and load it in the `__init__.py` of the `data.datamodules` module. Then you can pass the class path to the `class_path` argument in the configs and define all the arguments of the `__init__` function of the `DataModule`.

## The Config Files

The main config files are located in the `configs` directory. There is

* a `default.yaml` config file which defines some defaults (like early stopping on the trainer)
* an example `to_edit.yaml` file which you can use as inspiration
* a directory `data` with the datamodule configs
* a directory `eval` with the evaluation configs
* a directory `model` with the model configs
* a directory `slurm` with the slurm configs

If you have a new model or datamodule class, also add the respective config in the respective directory.

### `to_edit.yaml`

This is a central config file (you can rename `example_to_edit.yaml` as a starting point) which you can continuously edit for temporary config changes (which you don't want to "hard-code" in the other config files).

## General Project Configuration

The main project keys that you can define are

* `exp_name`: The name of the experiment. This value is used by the loggers to identify which project to log the experiment to.
* `machine`: A name you can give freely for the machine you started the experiment from.
* `run_name`: A name for the current experiment run. You don't have to set that because an automatic name will be generated from the model and data config that you pass. This is just an option if you want to name your run in a certain way.

It's best to put those values in your `to_edit.yaml` config (you only need to define `exp_name`).

## Trainer Configuration

The [trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) class is a central class of Pytorch Lightning which takes care of executing a training loop, etc. You can also configure its properties through the configs. E.g. in your `to_edit.yaml`

```yaml
trainer:
  limit_train_batches: 5
  gpus: 2
```

[All arguments of the trainer's init function](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api) are at your disposal!

Many trainer arguments are already predefined in the `defaults.yaml` file. For example, multiple callbacks are defined in this config file that automatically log histograms of the model weights and gradients, perform early stopping, etc. Since you would overwrite these callbacks if you defined a new `callback` attribute on the trainer in your `to_edit.yaml`, there is another key

```yaml
additional_callbacks:
```

in which you define a list of additional callbacks. The CLI will handle adding those to the existing trainer callbacks.

## Commands

To execute trainer commands there is the `commands` argument which defaults to `fit-test` (i.e. execute training/fitting and testing afterwards). You can define the commands you want to run either when running directly through Python:

    python src/cli.py --commands fit-test

or in MLflow:

    mlflow run ... -P commands=fit-test

or for both through the configs like:

```yaml
commands: fit-test
```

Available commmands are all functions of the trainer class: [`fit`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html), [`validate`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html), [`test`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html), [`predict`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) and [`tune`](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) together with their respective arguments. There is also the `eval` command which is not a trainer function an manually implemented.

Any combination of these, connected with `-`, is allowed as command. The commands will be executed in order, i.e.

    fit-test-evaluate

will excute first the fit function on the trainer, then test and then evaluate.

### Command Arguments

Since the functions are invoked on the trainer, you can pass the arguments to those functions also in the configs like so for example (note without a `trainer` key):

```yaml
validate:
    ckpt_path: ...
    verbose: True
```

This allows you to use a certain checkpoint through `ckpt_path`. Click on the functions to see which arguments you can define.

## Optimizers and Learning Rate Schedulers

The optimizer(s) and learning rate scheduler(s) can also be defined through the configs. For this, the keys `model.init_args.optimizer_init` and `model.init_args.lr_scheduler_init` are reserved. The class `AbstractModel` has a function called `configure_optimizers` which offers a default implementation to instantiate the optimizers and learning rate schedulers but if it doesn't fit your needs you can overwrite it.

You can checkout [Pytorch Lightning's docs on optimization](https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html), they provide more detail on what is to return in that function. It's also possible to manually optimize the model using the optimizers constructed here - that is also described in the Pytorch Lightning optimization docs.

You can use `instantiation_util.py` in the `utils` directory, to instantiate optimizers and lr schedulers in the `configure_optimizers` of your model (checkout `AbstractModel` for an example).

An example configuration can look like this:

```yaml
model:
    init_args:
        optimizer_init:
            class_path: torch.optim.SGD
            init_args:
                lr: 1.0e-3
                momentum: 0.9
                weight_decay: 0.01
        lr_scheduler_init:
            class_path: torch.optim.lr_scheduler.StepLR
            init_args:
                step_size: 5
                gamma: 0.1
```

Typically, some default optimization procedure is defined on the model configs in `configs/model` but you can overwrite them in your `to_edit.yaml` config like above.

Using the function `instantiate_lr_schedulers` of `instantiation_util.py`, you have multiple combination options:

1. You have 1 optimizer and 1 lr scheduler.
2. You have multiple optimizers but 1 lr scheduler. The scheduler will be instantiated for each optimizer.
3. You have multiple optimizers and multiple lr schedulers. In this case, either the number of optimizers has to match the number of lr schedulers. Then, either the first scheduler will be applied to the first optimizer and so on, or you provide the key `name` directly on the `optimizer_init` (same level as `class_path`) and `optimizer_name` on the `lr_scheduler_init` to match the schedluers and optimizers in a differnt order - *note* that in this case you need to provide names for all optimizers and schedulers because otherwise it would be ambiguous how to pair them. If the number of optimizers doesn't match the number of schedluers, you need to provide names on the schedulers which are also defined on the optimizers. For example:

```yaml
model:
    init_args:
        optimizer_init:
            - class_path: torch.optim.SGD
              name: optim1
              init_args:
                    lr: 1.0e-3
            - class_path: torch.optim.SGD
              name: optim2
              init_args:
                    lr: 1.0e-4
            - class_path: torch.optim.SGD
              init_args:
                    lr: 1.0e-2
        lr_scheduler_init:
            - class_path: torch.optim.lr_scheduler.StepLR
              optimizer_name: optim1
              init_args:
                    step_size: 5
            - class_path: torch.optim.lr_scheduler.StepLR
              optimizer_name: optim2
              init_args:
                    step_size: 3
```

## Transforms

Data transforms which can be used by the `DataModules` can also be defined in the config of the respective `DataModule` and overwritten from your `to_edit.yaml` config. For this, there is a function in `instantiation_util.py` which is called by the `AbstractDataModule` to instantiate the tranforms:

```python
instantiate_transforms_tree(caller, transform_tree: Any)
```

If you call this function manually, `caller` should be the `DataModule` you call this function from, so that transforms can access its attributes. This is possible since `instantiate_transforms_tree` will evaluate any expression of the form like `$self.img_size`.

It allows to nest dicts and lists and as soon as you provide a `class_path` entry will try to instantiate the respective class while also going over the `init_args` and checking if there is any class to instantiate in there. An exmple config could look like this:

```yaml
data:
    class_path: ...
    init_args:
        transforms:
            train:
                context:
                    class_path: torchvision.transforms.Compose
                    init_args:
                        transforms:
                            - class_path: torchvision.transforms.ToPILImage
                            - class_path: torchvision.transforms.RandomHorizontalFlip
                                init_args:
                                p: 0.3
```

This would put the `Compose` in `transforms['train']['context']` of the `DataModule`.

## Linking Arguments

It's possible to link arguments, e.g. `img_size` in the model and the datamodule. This can come in pretty handy as this way you have to define such a value only once. Checkout the [docs](https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html#argument-linking) for more details.

---
# Implementation

## Models

### Model Classes

The base class for all models is [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html). It defines various functions the trainer automatically calls during training, validation, etc. An example for one such function is `on_training_epoch_end` which you can overwrite if you want to execute some code after a training epoch.

`AbstractModel` is the base class for all models in this project. It inherits from `LightningModule` and defines a common `generic_step` function which will be called during a training, validation and testing step. If this `generic_step` doesn't work for you, overwrite it.

Next to `AbstractModel`, there are some other base classes defining common attributes which you can inhert form. For example

```python
class ImageClassificationModel(ImageModel, ClassificationModel)
```

has access to an image size, a mean and standard deviation, but also to a number of classes and logs a confusion matrix of the validation and test data.

Some key model functions:

| Function Name                       | Explanation                                                                                                                                       |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `configure_optimizers`                | Configuration of optimizers. Defining which parametes of the model to assign to which optimizer, etc.                                             |
| `_setup_losses`                       | Configuration of losses as ModuleDicts (`_train_losses`, `_val_losses`, `_test_losses`)                                                           |
| `_setup_metrics`                      | Configuration of metrics as ModuleDicts (`_train_metrics`, `_val_metrics`, `_test_metrics`)                                                       |
| `extract_inputs_from_batch`           | Extracts the inputs the network needs from the current batch. This allows reusing the same datamodule for multiple models.                        |
| `extract_targets_from_batch`          | Same as `extract_inputs_from_batch` but for targets                                                                                                 |
| `sanitize_outputs`                    | Detach outputs from the gradient graph. The model might return a complex structure of outputs where it's not directly clear how to sanitize them. |
| `extract_outputs_relevant_for_loss`   | Extract the outputs from the outputs dict of the model that you need for a certain loss.                                                          |
| `extract_targets_relevant_for_loss`   | Same as `extract_outputs_relevant_for_loss but` for targets                                                                                         |
| `extract_outputs_relevant_for_metric` | Extract outputs from the outputs dict of the model that you need for a certain metric.                                                            |
| `extract_targets_relevant_for_metric` | Same as `extract_outputs_relevant_for_metric` but for targets                                                                                       |

If you're implementing a rather simple network, the default implementation in `AbstractModel` might suffice but you can overwrite those functions if it doesn't. The `AbstractModel` uses those functions in `generic_step`.

### Backbones

To define backbones (e.g. feature extraction nets), you can use the key `backbone` in the model config. It can be

* Only a string, e.g. `resnet34`, which `backbone_load.py` will use to identify and load the respective backbone. The load will use ImageNet pretrained weights and add a new attribute `features` on the model object so that you don't have to change your code depending on the model type.
* A list of strings, which will all be instantiated by the backbone loader.
* Or, the upper two but instead you define the backbone as a dict with a `class_path` and (optional) `init_args` entry. This will directly instantiate them as you define them, avoiding having to use the backbone loader and giving you access to all availalbe pytorch ready-made networks.

### Implemented Models

* Simple classifier using various backbones
* Put-in-Context
* MANet
* CAERNet
* GLAMORNet
* SimCLR classifier

## Data

A goal of the project is to provide reproducible resluts. Therefore, we put all commands that are necessary to unpack/prepare/... a dataset in a Python class. That's what datamodule are for. 

### DataModules

A [`DataModule`](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html) contains the code to

* automatically download the dataset
* automatically extract/crop/... the downloaded dataset
* constructs the actual pytorch dataset to use in the code

The trainer will call the functions of the datamodule before starting in the following order:

1. `prepare_data`: Put some simple checks here if the dataset needs to be downloaded/unpacked/... again. If so, download the dataset and perform the necessary processing steps. This makes producing the dataset reproducible from scratch.
2. `setup`: Put the construction of the training, validation and test dataset in here.
3. `{train, val, test}_dataloader`: This function is used by the trainer to get the respective dataloader. There is a default implementation in `AbstractDataModule`, so unless you do something special don't worry about those.

The base class for all `DataModules` is `AbstractDataModule`. There are also `ImageDataModule` and `ClassificationDataModule` which provide some common attributes. You can inhert from both classes together to get an image-classifciation `DataModule`. Like in the model case, you can for example do

```python
class ImageClassificationDataModule(ClassificationDataModule, ImageDataModule)
```

## Trainer

The [trainer](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) class is a central class in Pytorch Lightning. It takes a model and a datamodule to perform training (called fitting), validation, testing, and so on. It can take care of distributed training, limiting the number of computed batches, early stopping and various other things.

You can access the trainer from the model through

```python
self.trainer
```

---
# Logging

The supported loggers are stored in `src/loggers`. Logging happens somewhat automatically through Pytorch Lightning. Everything a model returns in its training/validation/... step is automatically logged by the trainer using the passed loggers. Whenever you need to, you can manually log something using

```python
for logger in trainer.loggers
```

You have access to the trainer from the model (see above).

The currently implemented loggers are:

## MLflow

The [MLflow](https://www.mlflow.org/docs/latest/tracking.html) logger takes care of logging things either locally or remotely, in both ways under your full control.

### Setup

To use it, you need to set the `MLFLOW_TRACKING_URI` in your environment variables. Most likeley, you'll use MLflow for local tracking (so that you can directly inspect images, etc.) but it also supports remote tracking when you have a server at your disposal.

For local tracking, you can set the tracking URI e.g. to

    export MLFLOW_TRACKING_URI=/home/[username]/projects/facial-expression-recognitions/experiments

### Local Directories

This will create the project logging structure in the directory experiments. In the experiments dir, for each new experiment (which you can "create" by setting a new `exp_name` in your configs) a directory is created. In this directory, the experiment will be stored in a directory named using `run_name` and a random identifier consisting of an adjective + animal name for easy identification.

MLflow creates the following directories

* tags: Tags of the experiment
* params: Hyperparameters
* metrics: All metric output
* artifacts: Logs the full source code, checkpoints, the full config, etc. When you log an image using `log_image` or a figure using `log_figure`, it goes into an `imgs` directory in the `artifacts` directory.

### Config Logging and Reusing

The full config gets logged to `run_dir/artifacts/configs`. When you run the code and provide the `run_id` of the run, the config will get loaded. You can overwrite parts of it by still providing `--model`, etc., but you can omit these if you don't need to change any arguments.

If you reuse an experiment run and run something again, the new config will be logged using a timestemp. This is important as only the very first config that was logged will be loaded. E.g., when the logged config directory looks like this

```
config.yaml <-- Only this one will be loaded when run_id is defined
config_[timestemp1].yaml
config_[timestemp2].yaml
```

### MLflow UI

You can launch the MLflow UI to view your local experiments. For this, go to the experiments directory and type `mlflow ui`. If you're not on your local machine but e.g. on the cluster, you can checkout `run_mlflow_ui_server.sh` in the bash directory to see how to forward the content of the UI to your machine.

## CometML

The CometML logger takes care of logging things to [comet.ml](www.comet.ml) which is similar to wandb, neptune.ai, and so on. To use it, you need to set the `COMET_API_KEY` in your environment variables - put it in your `.bashrc`. You can find the key in the settings on comet.ml.

CometML's run ID and MLflow's run ID are not the same because CometML assigns one automatically and we have no influence on this. But to make it easier to identify experiments, I chose to give run IDs that use the configs' names as well as this well known pattern of adjective + animal name. The code relies on the run IDs from MLflow, i.e. when you want to e.g. continue an experiment, you need the run ID that is printed in the terminal.

The MLflow run ID is also logged to CometML which allows you to identify the local run ID online on the comet.ml website.

### Debugging CometML

In case you need more CometML output because something does not work, go to the CometML Logger - `cometml.py` in `loggers` directory - and comment out the line

    self.experiment.display_summary_level = 0

which reduces the amount of output by CometML.

## Synching between MLflow and CometML

Since you can delete local experiments from MLflow without CometML noticing, and vice versa, there are some functions availalbe to synchornize the two again. Checkout the `MLproject_entrypoints.yaml` file to see which exactly.

## Your Own Logger

Implementing your own logger is pretty easy. Pytorch Lightning offers a lot of [logging integrations](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) already. You just need to write a wrapper around them, in the style of the mflowlogger and cometmllogger you can find in `src/loggers`.

## Using Loggers

You can simply define the loggers in your `to_edit.yaml` like this:

```yaml
trainer:
  logger:
  - class_path: loggers.CometLogger
  - class_path: loggers.MLFlowLogger
```

which will use both loggers for logging.

---
# Training

Training is handled by the trainer automatically. For training to happen, you need to provide `fit` as part of the `commands` argument. Well, you don't need to since `commands` defauls to `fit-test` anyways, i.e. the model is first fitted/trained and then tested afterwards. This ensures that the two always happen together.

## Training from Scratch

For training from scratch you don't have to do much. A simple

    mlflow run MLproject_conda -e main -P model=configs/model... -P data=configs/data... -P config=configs/to_edit.yaml

will start training and testing. Since there's an early stopping callback on the trainer by default you don't have to set a number of epochs since it monitors `avg_val_loss` and terminates training when this doesn't decrease for some epochs.

You can define some pretrained weights for the backbone to load by using

```yaml
model:
    init_args:
        backbone_weights_path: ...
```

If you defined multiple backbones you need to define multiple weights paths here.

## Continuing Training/Weight Loading

To continue training of an experiment you can set the `run_id` in your `to_edit.yaml` to the respective ID and in `fit` function configuration refer to the checkpoint you want to load. When launching an experiment, the run's ID is printing in the beginning. It's also logged as a parameter.

**Setting `run_id` will automatically look for logged configs and load them, i.e. you can most likely omit `--model`, `--data`, and so on and only need to provide your `to_edit.yaml` config.**

So, your `to_edit.yaml` could look like this

```yaml
fit:
    ckpt_path: $ckpt_dir/epoch-04=val-1.27.ckpt
```

where `$ckpt_dir` will be resolved to the run's checkpoint directory. You can also provide an absolute/relative path without `$ckpt_dir`. And your command could be this

    mlflow run MLproject_conda -e main -P run_id=[the run id] -P config=configs/to_edit.yaml

and that's it. This will launch using the default `fit-test` command and load the checkpoint you provided.

To find the checkpoint you want to use you need the run ID and need to know which experiment the run is in. Then you go to the experiments directory -> [run directory of run with the ID] -> artifacts -> ckpts.

## Manual Training/Testing

If you don't want to use the trainer to train for some reason, don't forget to call `eval()` and `train()` on the model.

---
# Testing & Evaluation

Testing is usually also carried out in the end of training since the default command is `fit-test`. That's why you should leave `fit.ckpt_path` set to `best`, this will use the best model weights after training automatically.

## Manual Running of Testing

To test manually, you first need to find out the `run_id`. When launching an experiment, the run's ID is printing in the beginning. It's also logged as a parameter.

You can then

* Set `run_id` in your configs to the run id - this will reuse the run and its directories.
* Load the checkpoint for the respective function through e.g. `fit.ckpt_path: $ckpt_dir/epoch=04-avg_val_loss=1.56.ckpt`. `$ckpt_dir` will be replaced by the checkpoints directory of the run. You can also provide an absolute or relative path without `$ckpt_dir`.

Providing the `run_id` means that the code will look for logged configs and will try to load them. This means that you can omit `--model`, `--data`, and so but you can provide them to overwrite something in the logged config.

To run testing you can run one of the following commands

* python src/cli.py --run_id= [--model= ....] --commands=test
* mlflow run [environment file] -e test -P run_id= [-P model=...]
* mlflow run [environment file] -e main -P run_id= [-P model=...] -P commands=test

`[-P model=...]` is supposed to mean that you can leave those parts out.

## Evaluation

Next to testing there also exist certain evaluation functions in the `src/evaluation` directory. You can run those also through the CLI. Similarly to manual testing, you need to

* Provide `run_id` if you want to reuse an existing run (e.g. to load a checkpoint), some evaluation functions like `inspect_batch_inputs` do not need an existing run and will create a new run if `run_id` is not provided. You can provide it via the configs or by `--run_id` or `-P run_id=`.
*  Provide `eval_func` and set it to the name of the evaluation function you want to run.
* Provide the arguments the evaluation function expects (checkout the function signature) **except** for the `model`, `datamodule` and `trainer` arguments - those are automatically provided by the CLI. E.g. in your `to_edit.yaml`:

```yaml
eval:
    ckpt_path: ...
```

The configs in `configs/eval` show which functions are avaiable and which arguments they expect. So, the command to execute evaluation could look like this

    mlflow run MLproject_conda -e eval -P run_id=[run id] -P eval=... -P config=configs/to_edit.yaml

Note that `-P eval=...` internally gets translated to a `--config` argument. You could also call it like this

    python src/main.py --run_id [run_id] --config configs/eval/... --config configs/to_edit.yaml

### Implementing Your Own Evaluation Function

Implementing your own evaluation function is pretty easy. You need to create a python file in `src/evaluation` with the name of the function you want to create, e.g. `inspect_batch_inputs` and inside a function using the same name must be present. The function must look like this

```python
def inspect_batch_inputs(model, datamodule, trainer, [any additonal args])
```

`model`, `datamodule` and `trainer` will be provided by the CLI. You can add an arbitrary number of arguments (and add defaults if you'd like). You can then define them through the `eval` key in the configuration like you're used to.

Then you can also create a config file in `configs/eval` to provide easy access and a template for your function arguments like so

```yaml
eval_func: examine_incorrectly_classified_images
eval:
  ckpt_path: null # need to set in to_edit.yaml
```

### Availble Common Evaluation Functions

| Function Name        | Explanation                       |
|----------------------|-----------------------------------|
| inspect_batch_inputs | Logs image batch inputs as images |
| examine_incorrectly_classified_images                     | Logs images which are incorrectly classified                                  |

---
# Docker

## Getting the Docker image

MLflow will automatically retrieve the approriate image (i.e. the one defined in `MLproject_docker`) but you also build it yourself:

        docker build -t florianblume/facial-expression-recognition:latest -f docker/main/Dockerfile --no-cache .

## Running the Code in Docker

You can of course run the code in Docker. The easiest way is to use MLflow to do so as it takes care of mounting all necessary directories as well as provides environment variables, etc. To run the code in Docker simply do

    mlflow run MLproject_docker -e main -P model=... -A gpus=all -A ipc=host

The last parts `-A gpus=all` and `-A ipc=host` are important as they will be translated to parts of the Docker command that make GPUs available and use the host for inter-process communication (IPC). Without the latter the data loading processes will crash.

When MLflow launches the code you can also see the full command that was used. You can copy it and execute it yourself also directly (same actually also holds when using Anaconda).

## Environment Variables

In `MLproject_docker` you'll see some environment variables already defined which are necessary to launch the code. If you need additional ones add them here, they will be loaded from the host system.

When you look at the volumes, you'll see that `MLFLOW_TRACKING_URI` is not only an environment variable but also used as a volume name which will be mounted. MLflow interprets volume names with a leading `$` which allows you to e.g. use `$PWD` to refre to the working directory.

The `$TORCH_HOME` environment variable is useful since pytorch pretrained weights will be downloaded to that directory. If this environment variable is present, MLflow will automatically tell Pytorch about it when launching the code through Docker. Since `TORCH_HOME` is accessed to store data, it is also mounted as a volume.

## Folders to Mount

If you need additional folders to be mounted, make sure the non-root user in the Docker image has read-write rights:

    chmod 766 /path/to/the/folder/to/mount

Otherwise Docker will not be able to access it properly.

**Note:** This gives a lot of rights to many users for that folder. If that's an issue you'll have to look for a solution yourself ;)

---
# Slurm

## Scheduling

Slurm can be configured similarly to the project in general. There is the special `slurm` argument which is a dictionary containing all Slurm related stuff. To **schedule** your run via Slurm simply do

    mlflow run MLproject_slurm -e slurm -P model=... -P data=... -P config=... -P slurm=...

This creates a new experiment run, copies the configs to the run directory (as Slurm is sometimes pretty slow to launch so you don't accidentially edit a config before the run starts), and schedules the run in Slurm.

Since the entrypoint is `-e slurm` you can unfortunately not execute `-e test` or `-e eval` for example. Instead, you'll have to pass those using the `-P command` parameter.

Of course the same also works in plain Python:

    python src/slurm.py --model ...

## Slurm Configs

For `slurm=...` you can select one of the slurm configs in `configs/slurm`. The default config `configs/slurm/defaul.yaml` will get loaded automatically. Since Slurm is also configured through configs, you can overwrite config values e.g. in your `to_edit.yaml` config.

Slurm logs will be automatically written to a `slurm` directory in the run's artifacts directory.

## Continuing Runs in Slurm

To continue an experiment run just provide the `run_id` key as you are used to. This will also load the existing config, i.e. you can omit the model, data, etc. configs.

# Issues, Common Pitfalls & Debugging

## Configuration Issues

In case you need to debug jsonargparse since there are config issues you can set
`export JSONARGPARSE=1` (or to any value) and you should see a more extensive debug log.

When running the code, the following error occurs:

---

```
cli.py: error: Configuration check failed :: Parser key "model": Type <class 'models.abstract_model.AbstractModel'> expects: a class path (str); or a dict with a class_path entry; or a dict with init_args (if class path given previously). Got "{'init_args': {'batch_size': 32}}".
```

This might be because your `to_edit.yaml` looks like this

```yaml
model:
    init_args:
        batch_size: 32
```

and you do not provide a `class_path` there. This is fine but the `class_path` has to come from some other config. Thus, possible solutions are

* To check for mistakes in the path to the model config (e.g. typed `class_resnet.yaml` instead of `class_resnet34.yaml`, the former one doesn't exist - unless you created it of course).
* You completely forgot to provide a model config - add one using `--model` for running directly through Python or `-P model=...` when using MLflow.

---

```
cli.py: error: Parser key "model": Problem with given class_path "models.ImageClassificationModel":
- Configuration check failed :: Key "img_size" is required but not included in config object or its value is None.
```

In general, this error occurs when the model, data, ... class expects a parameter (does not have a default value) and you didn't provide it. For example, the model configs don't have an image size on purpose so that you remember ot define it because image size is varied often. The error above tells you that you forogt to set it in your `to_edit.yaml`, so the solution could be to put

```yaml
model:
    init_args:
        img_size: [100, 150]
```

in the config.

---

```
cli.py: error: Problems parsing config :: mapping values are not allowed in this context
```

There is a wrong indent in one of the configs, e.g.

```yaml
model:
    class_path: ...
     init_args: ...
```

The `init_args` are wrongly indented.

## MLflow run doesn not completely clear the video memory

Unkown issue which I have to investigate some time. Use

    for i in $(sudo lsof /dev/nvidia0 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done

to free the memory again. This kills all nvidia processes spawned by Python.

## Matrix Size Mismatch

* Mismatch in matrix size -> adjust num classes, batch size, ...
* Other issues

---
# Development Advice

## For deploying MLflow

* nodejs/npm --> `conda install nodejs -c conda-forge --repodata-fn=repodata.json`, then in `mlflow/server/js` execute `npm install && npm run build` then in one shell `mlflow ui` and in another `cd mlflow/server/js && npm start` and on the local machine `ssh -L localhost:5000:localhost:5000 florian@galilei`
* build wheel with `python setup.py bdist_wheel`

---
# Notes

Stuff that could become annoying because I forget I did it and then get mad it myself when I find out in the code again that I did it.

* comet ml save dir is set to none manually in before_instantiate_classes
* How do outputs look like in epoch_ends [num_train_steps, batch_size * model_output_size]
----
#Cite
If you use this work in your research or find it helpful, please cite:

    @misc{halawa2024multitask,
      title={Multi-Task Multi-Modal Self-Supervised Learning for Facial Expression Recognition}, 
      author={Marah Halawa and Florian Blume and Pia Bideau and Martin Maier and Rasha Abdel Rahman and Olaf Hellwich},
      year={2024},
      eprint={2404.10904},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}
