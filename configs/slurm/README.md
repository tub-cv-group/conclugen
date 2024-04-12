# Slurm

## Scheduling

Slurm can be configured similarly to the project in general. There is the special `slurm` argument which is a dictionary containing all Slurm related stuff. To **schedule** your run via Slurm simply do

    mlflow run MLproject_slurm -e slurm -P model=... -P data=... -P config=... -P slurm=...

This creates a new experiment run, copies the configs to the run directory (as Slurm is sometimes pretty slow to launch so you don't accidentially edit a config before the run starts), and schedules the run in Slurm.

Of course the same also works in plain Python:

    python src/slurm.py --model ...

## Slurm Configs

For `slurm=...` you can select one of the slurm configs in `configs/slurm`. The default config `configs/slurm/defaul.yaml` will get loaded automatically. Since Slurm is also configured through configs, you can overwrite config values e.g. in your `to_edit.yaml` config.

Slurm logs will be automatically written to a `slurm` directory in the run's artifacts directory.

## Continuing Runs in Slurm

To continue an experiment run just provide the `run_id` key as you are used to. This will also load the existing config, i.e. you can omit the model, data, etc. configs.