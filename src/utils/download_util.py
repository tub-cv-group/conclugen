import os
from nlpaug.util.file.download import DownloadUtil
from utils import constants as C


def download_word2vec_model():
    model_dir = dict(os.environ).get(C.ENV_WORD2VEC_MODEL_DIR, 'models/word2vec')
    model_file_path = os.path.join(model_dir, 'GoogleNews-vectors-negative300.bin')
    if not os.path.exists(model_file_path):
        os.makedirs(model_dir, exist_ok=True)
        DownloadUtil.download_word2vec(dest_dir=model_dir) # Download word2vec model
    return model_file_path
