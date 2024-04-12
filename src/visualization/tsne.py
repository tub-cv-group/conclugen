import argparse
from tsnecuda import TSNE
import glob
import numpy as np
#from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import tqdm
import sys
 
parser = argparse.ArgumentParser(description="Feature Visualization script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-d", "--dataset",  help="dataset")
parser.add_argument("-vm", "--vismethod",  help="visualization method like TSNE or UMAP")
args = parser.parse_args()


if args.dataset== 'voxceleb2':
    pass
else:
    raise Exception("wrong dataset") 
modes = ['frames_2d', 'frames_3d', 'texts', 'audio_spectrograms_40']
embeddings = []
fig, axs = plt.subplots(2,2)

for index, mode in enumerate(modes):
    paths = glob.glob(f"/shared/datasets/project08/processed/voxceleb2/features/{mode}/train/*.npy")
    dim = np.load(paths[0]).shape[0]
    amount = len(paths)
    print('dimension: ', dim)
    print('datapoints: ', amount)
    if not (mode == 'texts'):
        bins = range(amount)
        # bins = np.array([range(100000)]+ [range(amount//2-50000, amount//2+50000)]+[range(amount-100000,amount)])
        # bins = bins.flatten()
    else:
        bins = np.array([range(10000)]+ [range(amount//2-5000, amount//2+5000)]+[range(amount-10000,amount)])
        bins = bins.flatten()    
    setbins = set(bins)
    print('len(setbins): ',len(setbins))
    print("len(bins): ", len(bins))
    features = np.empty((len(bins),dim))
    for i, bin in enumerate(bins):
        features[i] = np.load(paths[bin])
    if args.vismethod == "TSNE":
        embeddings = ((TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(features)))
    if args.vismethod == "UMAP":
        embeddings= (umap.UMAP().fit_transform(features))
    axs[index // 2, index % 2].scatter(embeddings[ :, 0], embeddings[:, 1], s = 1)
    axs[index // 2, index % 2].set_title(mode)
    axs[index // 2, index % 2].set_xticklabels([])
    axs[index // 2, index % 2].set_yticklabels([])


fig.set_figheight(20)
fig.set_figwidth(20)
fig.suptitle(f"{args.vismethod}") 
fig.savefig(f"{args.vismethod}.svg")

