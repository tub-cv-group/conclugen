# [Docker](www.docker.io)

You can also use the Docker images found at [Docker hub](ymousano/ssrl-fer-study:latest) or use the Dockerfiles in the `docker` directory to build it yourself. There's also a script in the `bash` directory which contains the command to build the Docker image (be careful, it is configured to delete the cache so it will always build the Dockerfile). The command to build it is

    docker build -t ymousano/ssrl-fer-study:latest -f docker/main/Dockerfile --no-cache .

When you want to use Docker you might have to create specific folders before running the code and mount them manually/through the entrypoints file since Docker has no direct access to the underlying filesystem. I.e., create the folders on your local machine and modify access rights like so

    chmod 766 /path/to/the/folder/to/mount

## Getting the Docker image

MLflow will automatically retrieve the approriate image (i.e. the one defined in `MLproject_docker`) but you also build it yourself:

        docker build -t ymousano/ssrl-fer-study:latest -f docker/main/Dockerfile --no-cache .

## Running the Code in Docker

You can of course run the code in Docker. The easiest way is to use MLflow to do so as it takes care of mounting all necessary directories as well as provides environment variables, etc. To run the code in Docker simply do

    mlflow run MLproject_docker -e main -P model=... -A gpus=all -A ipc=host

The last parts `-A gpus=all` and `-A ipc=host` are important as they will be translated to parts of the Docker command that make GPUs available and use the host for inter-process communication (IPC). Without the latter the data loading processes will crash.

When MLflow launches the code you can also see the full command that was used. You can copy it and execute it yourself also directly (same actually also holds when using Anaconda).

## Environment Variables

In `MLproject_docker` you'll see some environment variables already defined which are necessary to launch the code. If you need additional ones add them here, they will be loaded from the host system.

When you look at the volumes, you'll see that `MLFLOW_TRACKING_URI` is not only an environment variable but also used as a volume name which will be mounted. MLflow interprets volume names with a leading `$` which allows you to e.g. use `$PWD` to refre to the working directory.

The `$TORCH_HOME` environment variable is useful since pytorch pretrained weights will be downloaded to that directory. If this environment variable is present, MLflow will automatically tell Pytorch about it when launching the code through Docker. Since `TORCH_HOME` is accessed to store data, it is also mounted as a volume.