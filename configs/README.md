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

[All arguments of the trainer's init function](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api) are at your disposal! Many trainer arguments are already predefined in the `defaults.yaml` file.

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