model_name=$1;
dataset_name=$2;

# If model_name or dataset_name are empty, print usage
if [ -z $model_name ] || [ -z $dataset_name ]; then
    echo "Usage: ./run.sh <model_name> <dataset_name>"
    echo "model_name one of: conclugen, generative, simclr, contrast_cluster, contrast_generative, contrast_only, contrast_only_video_audio, contrast_only_video_text"
    echo "dataset_name one of: caer, meld, mosei, ravdess"
    exit 1
fi

echo "Running model: $model_name on dataset: $dataset_name"

conclu_caer="https://www.comet.com/api/asset/download?assetId=09466805a99f4febbec7d2acf95f733f&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
conclu_meld="https://www.comet.com/api/asset/download?assetId=c45fcfb0ffd649f7bed393f47c3c75ab&experimentKey=4acd44d58a714898bad65443e77c61ac"
conclu_mosei="https://www.comet.com/api/asset/download?assetId=dc8a3b7962d44f19820f43ef96cd4619&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

contrast_cluster_caer="https://www.comet.com/api/asset/download?assetId=98a4f631864b4b2f9e3730014dfbdded&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_cluster_meld="https://www.comet.com/api/asset/download?assetId=d237fe78815c4a1080eab93a4beedd84&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_cluster_mosei="https://www.comet.com/api/asset/download?assetId=65316bbc2961418ea0b074cb2944f302&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

contrast_generative_caer="https://www.comet.com/api/asset/download?assetId=85c101a71459493e9933ccab65d13d11&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_generative_meld="https://www.comet.com/api/asset/download?assetId=011cc33af4ca455790fdc738e2fceab2&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_generative_mosei="https://www.comet.com/api/asset/download?assetId=3f08e6cdf46c42b1b97fc8a769cbb3c0&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

contrast_only_caer="https://www.comet.com/api/asset/download?assetId=120a11bfb7bd45ec8a49ba18bf891b6d&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_meld="https://www.comet.com/api/asset/download?assetId=2066a246a3e547728fdba849d820e4ee&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_mosei="https://www.comet.com/api/asset/download?assetId=d196dea4484e4e17b3209d099511ed8c&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

contrast_only_video_audio_caer="https://www.comet.com/api/asset/download?assetId=e0b56a9eac8942fa9607071b23d8eb0d&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_video_audio_meld="https://www.comet.com/api/asset/download?assetId=f0ff2a1f84a84ea397e00f9d6794cc19&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_video_audio_mosei="https://www.comet.com/api/asset/download?assetId=74e48c1b64784706b253d65a778eb6ea&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

contrast_only_video_text_caer="https://www.comet.com/api/asset/download?assetId=edc9b8de4c27478ca8e4c6651515aeb8&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_video_text_meld="https://www.comet.com/api/asset/download?assetId=e7419321d7d74ab98b9a43f6c7f32b90&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
contrast_only_video_text_mosei="https://www.comet.com/api/asset/download?assetId=970a885a471142c1ae891870b49b18fd&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

generative_caer="https://www.comet.com/api/asset/download?assetId=92d207bf14094805aada9b22ee0e795f&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
generative_meld="https://www.comet.com/api/asset/download?assetId=a0a444878d5d4ea083411a6ab69e93e1&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
generative_mosei="https://www.comet.com/api/asset/download?assetId=b42303adb7b340b79cd33bbdff1d3b3e&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

simclr_caer="https://www.comet.com/api/asset/download?assetId=5d63c242f75a4fc5a85903125bc7fb40&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
simclr_meld="https://www.comet.com/api/asset/download?assetId=4dcf1f101fcc42b6a2777ba8d750d264&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"
simclr_mosei="https://www.comet.com/api/asset/download?assetId=5d63c242f75a4fc5a85903125bc7fb40&experimentKey=69cd80f8c5ae426a8d29e04182b8083e"

# Select the model checkpoint based on the model and dataset name
if [ $model_name == "conclugen" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$conclu_caer
    elif [ $dataset_name == "meld" ]; then
        model=$conclu_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$conclu_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "generative" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$generative_caer
    elif [ $dataset_name == "meld" ]; then
        model=$generative_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$generative_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "simclr" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$simclr_caer
    elif [ $dataset_name == "meld" ]; then
        model=$simclr_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$simclr_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "contrast_cluster" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$contrast_cluster_caer
    elif [ $dataset_name == "meld" ]; then
        model=$contrast_cluster_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$contrast_cluster_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "contrast_generative" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$contrast_generative_caer
    elif [ $dataset_name == "meld" ]; then
        model=$contrast_generative_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$contrast_generative_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "contrast_only" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$contrast_only_caer
    elif [ $dataset_name == "meld" ]; then
        model=$contrast_only_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$contrast_only_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "contrast_only_video_audio" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$contrast_only_video_audio_caer
    elif [ $dataset_name == "meld" ]; then
        model=$contrast_only_video_audio_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$contrast_only_video_audio_mosei
    else
        echo "Model not found"
        exit 1
    fi
elif [ $model_name == "contrast_only_video_text" ]; then
    if [ $dataset_name == "caer" ]; then
        model=$contrast_only_video_text_caer
    elif [ $dataset_name == "meld" ]; then
        model=$contrast_only_video_text_meld
    elif [ $dataset_name == "mosei" ]; then
        model=$contrast_only_video_text_mosei
    else
        echo "Model not found"
        exit 1
    fi
else
    echo "Model not found"
    exit 1
fi

type="single"

caer="https://www.comet.com/api/artifacts/version/download?versionId=9902c08a-9486-45bf-94c3-92b77390367f"
meld="https://www.comet.com/api/artifacts/version/download?versionId=bedc3832-d411-44ad-bd1b-70c659d71e02"
mosei="https://www.comet.com/api/artifacts/version/download?versionId=40a46ebb-204d-4e4c-b2a6-493290db0fb6"
ravdess="https://www.comet.com/api/artifacts/version/download?versionId=df691105-f25f-4745-9d30-560b44c173cc"

# Select url based on dataset_name
if [ $dataset_name == "caer" ]; then
    dataset=$caer
elif [ $dataset_name == "meld" ]; then
    dataset=$meld
elif [ $dataset_name == "mosei" ]; then
    dataset=$mosei
    type="multi"
elif [ $dataset_name == "ravdess" ]; then
    dataset=$ravdess
else
    echo "Dataset name not found"
    exit 1
fi

mlruns_dir="mlruns"
mkdir -p $mlruns_dir
hugginface_cache_dir="cache/hugginface/transformers"
mkdir -p $hugginface_cache_dir
torch_home_dir="cache/torch"
mkdir -p $torch_home_dir

echo "Using model weights from $model"
echo "Using dataset from $dataset"

echo "Running docker command"

if [ $model_name == "simclr" ]; then
    docker run --rm --workdir /mlflow/projects/code/ --gpus all --ipc host -v $PWD:/mlflow/projects/code -e MLFLOW_TRACKING_URI=$mlruns_dir -e \
    COMET_API_KEY=$COMET_API_KEY -e TORCH_HOME=$torch_home_dir -e TRANSFORMERS_CACHE=$hugginface_cache_dir \
    ymousano/ssrl-fer-study:latest "python src/mlflow_entrypoints.py mlflow_main --model=configs/model/simclr/simclr_3d_$type.yaml --data=configs/data/$dataset_name/${dataset_name}_cropped_precomp_augmented_frames_3d.yaml --config=configs/backbone/resnet3d101.yaml --config=configs/testing.yaml --model.init_args.model_weights_path '$model' --data.init_args.data_dir '$dataset' --commands=test"
else
    docker run --rm --workdir /mlflow/projects/code/ --gpus all --ipc host -v $PWD:/mlflow/projects/code -e MLFLOW_TRACKING_URI=$mlruns_dir -e \
    COMET_API_KEY=$COMET_API_KEY -e TORCH_HOME=$torch_home_dir -e TRANSFORMERS_CACHE=$hugginface_cache_dir \
    ymousano/ssrl-fer-study:latest "python src/mlflow_entrypoints.py mlflow_main --model=configs/model/conclu/concat/conclu_$type.yaml --data=configs/data/$dataset_name/${dataset_name}_cropped_precomp_augmented.yaml --config=configs/testing.yaml --strconfig=\"model.init_args.model_weights_path=$model;data.init_args.data_dir=$dataset\" --commands=test"
fi
