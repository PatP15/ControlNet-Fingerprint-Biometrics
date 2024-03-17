import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
import os
import math

sys.path.append('../')
sys.path.append('../directory_organization')

from trainer import *

from siamese_datasets import *
from fingerprint_dataset import *
from embedding_models import *

# from common_filepaths import DATA_FOLDER, SYNTHETIC_DATA_FOLDER
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



#diffusion model
# MODEL_PATH = '../FingerPrintMatchingModels/diffusion/finetune_embeddings_diffusion.pth'
# MODEL_PATH = '../FingerPrintMatchingModels/diffusion/model_embeddings_diffusion.pth'

# MODEL_PATH = '/data/puma_envs/FingerPrintMatchingModels/diffusion/model_embeddings_diffusion2.pth'
MODELS = ['../FingerPrintMatchingModels/diffusion/finetune_embeddings_diffusion2.pth', 
          '../FingerPrintMatchingModels/finetune_embeddings_printsgan2.pth',
          '../FingerPrintMatchingModels/imagenet/finetune_embeddings_imagenet.pth',
            '../FingerPrintMatchingModels/baseline/model_embeddings_baseline.pth'
          ]

#printsGAN
# MODEL_PATH = '../FingerPrintMatchingModels/finetune_embeddings_printsgan.pth'
# MODEL_PATH = '../FingerPrintMatchingModels/finetune_embeddings_printsgan2.pth'
# MODEL_PATH = '../FingerPrintMatchingModels/model_embeddings_printsgan.pth'
#imagenet
# MODEL_PATH = '../FingerPrintMatchingModels/imagenet/finetune_embeddings_imagenet.pth'

#baseline
# MODEL_PATH = '../FingerPrintMatchingModels/baseline/model_embeddings_baseline.pth'

SD301_PATH = '/home/puma/data/therealgabeguo/fingerprint_data/sd301_split'
# TEST_PATH = '/data/puma_envs/img_l2_feature_extractions/enhance_control_generated'
batch_size=16

#training_dataset = TripletDataset(FingerprintDataset(os.path.join(SYNTHETIC_DATA_FOLDER, 'train'), train=True))
#training_dataset = torch.utils.data.Subset(training_dataset, list(range(0, len(training_dataset), 5)))
#train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

#val_dataset = TripletDataset(FingerprintDataset(os.path.join(SYNTHETIC_DATA_FOLDER, 'val'), train=False))
#val_dataset = torch.utils.data.Subset(val_dataset, list(range(0, len(val_dataset), 5)))
#val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(FingerprintDataset(os.path.join(SD301_PATH, 'test'), train=False))
#test_dataset = torch.utils.data.Subset(test_dataset, list(range(0, len(test_dataset), 100)))
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# SHOW IMAGES
"""
import matplotlib.pyplot as plt
it = iter(val_dataloader)
for i in range(5):
    images, labels, filepaths = next(it)
    next_img = images[2][0]
    the_min = torch.min(next_img)
    the_max = torch.max(next_img)
    next_img = (next_img - the_min) / (the_max - the_min)
    print(next_img[0])
    plt.imshow(next_img.permute(1, 2, 0))
    plt.show()
"""
"""
Saves a ROC curve
"""
import matplotlib.pyplot as plt

def plot_roc_auc(fpr, tpr, dataset_name, weights_name, output_dir):
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'roc_curve_{}_{}.pdf'.format(\
        dataset_name, weights_name)))
    plt.savefig(os.path.join(output_dir, 'roc_curve__{}_{}.png'.format(\
        dataset_name, weights_name)))
    plt.clf(); plt.close()

    return
"""
Inputs: (_01_dist, _02_dist) are distance between anchor and (positive, negative), repsectively
Returns: (acccuracies, fpr, tpr, ROC AUC, threshold, welch_t, p_val)
- accuracies are at every possible threshold
- fpr is false positive rate at every possible threshold (padded with 0 and 1 at end)
- tpr is true positive rate at every possible threshold (padded with 0 and 1 at end)
- roc_auc is scalar: area under fpr (x-axis) vs tpr (y-axis) curve
- threshold is scalar: below this distance, fingerpritnts match; above, they don't match
- welch_t is value of Welch's two-sample t-test between same-person and diff-person pairs
- p-val is statistical significance
"""
def get_metrics(_01_dist, _02_dist):
    all_distances = _01_dist +_02_dist
    all_distances.sort()

    tp, fp, tn, fn = list(), list(), list(), list()
    acc = list()

    # try different thresholds
    for dist in all_distances:
        tp.append(len([x for x in _01_dist if x < dist]))
        tn.append(len([x for x in _02_dist if x >= dist]))
        fn.append(len(_01_dist) - tp[-1])
        fp.append(len(_02_dist) - tn[-1])

        acc.append((tp[-1] + tn[-1]) / len(all_distances))
    threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]

    # ROC AUC is FPR = FP / (FP + TN) (x-axis) vs TPR = TP / (TP + FN) (y-axis)
    fpr = [0] + [fp[i] / (fp[i] + tn[i]) for i in range(len(fp))] + [1]
    tpr = [0] + [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))] + [1]
    auc = sum([tpr[i] * (fpr[i] - fpr[i - 1]) for i in range(1, len(tpr))])

    assert auc >= 0 and auc <= 1

    for i in range(1, len(fpr)):
        assert fpr[i] >= fpr[i - 1]
        assert tpr[i] >= tpr[i - 1]


    return acc, fpr, tpr, auc, threshold

# Pre: parameters are 2 1D tensors
def euclideanDist(tensor1, tensor2):
    return (tensor1 - tensor2).pow(2).sum(0)

def run_test(MODEL_PATH, iteration):
    test_dataset = TripletDataset(FingerprintDataset(os.path.join(SD301_PATH, 'test'), train=False))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and load state
    embedder = EmbeddingNet().to('cuda:0')
    embedder.load_state_dict(torch.load(MODEL_PATH))
    embedder.eval()
    triplet_net = TripletNet(embedder)

    _01_dist = []
    _02_dist = []

    # Test loop
    for i in range(len(test_dataloader)):
        test_images, test_labels, test_filepaths = next(iter(test_dataloader))

        test_images = [item.to('cuda:0') for item in test_images]

        embeddings = [torch.reshape(e, (batch_size, e.size()[1])) for e in triplet_net(*test_images)]
        # len(embeddings) == 3 reprenting the following (anchor, pos, neg)
        # Each index in the list contains a tensor of size (batch size, embedding length)

        for batch_index in range(batch_size):
            _01_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[1][batch_index]).item())
            _02_dist.append(euclideanDist(embeddings[0][batch_index], embeddings[2][batch_index]).item())

            # process traits of these samples (same finger, same sensor)
            anchor_filename, pos_filename, neg_filename = \
                test_filepaths[0][batch_index], test_filepaths[1][batch_index], test_filepaths[2][batch_index]
            anchor_filename, pos_filename, neg_filename = anchor_filename.split('/')[-1], pos_filename.split('/')[-1], neg_filename.split('/')[-1]

            # print(anchor_filename, pos_filename, neg_filename)

        if i % 40 == 0:
            print('Batch {} out of {}'.format(i, len(test_dataloader)))
            print('\taverage squared L2 distance between positive pairs:', np.mean(np.array(_01_dist)))
            print('\taverage squared L2 distance between negative pairs:', np.mean(np.array(_02_dist)))

    accs, fpr, tpr, auc, threshold = get_metrics(_01_dist, _02_dist)
    # FIND THRESHOLDS

    print('best accuracy:', max(accs))
    # threshold = all_distances[max(range(len(acc)), key=acc.__getitem__)]

    print('number of testing positive pairs:', len(_01_dist))
    print('number of testing negative pairs:', len(_02_dist))

    print('average squared L2 distance between positive pairs:', np.mean(_01_dist))
    print('std of  squared L2 distance between positive pairs:', np.std(_01_dist))
    print('average squared L2 distance between negative pairs:', np.mean(_02_dist))
    print('std of  squared L2 distance between negative pairs:', np.std(_02_dist))

    from datetime import datetime
    datetime_str = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    model_name = MODEL_PATH.split('/')[-1].split('.')[0]
    print(model_name)
    output_dir = '/home/puma/results'
    dataset_name = "SD301/test"

    plot_roc_auc(fpr=fpr, tpr=tpr, dataset_name=dataset_name, weights_name=model_name + f"_iteration_{iteration}", output_dir=output_dir)

    with open(f'/home/puma/results/{model_name}_iteration_{iteration}.txt', 'w') as fout:
        fout.write(f'Iteration: {iteration}\n')
        fout.write('average squared L2 distance between positive pairs: {}\n'.format(np.mean(_01_dist)))
        fout.write('std of squared L2 distance between positive pairs: {}\n'.format(np.std(_01_dist)))
        fout.write('average squared L2 distance between negative pairs: {}\n'.format(np.mean(_02_dist)))
        fout.write('std of squared L2 distance between negative pairs: {}\n'.format(np.std(_02_dist)))
        fout.write('auc: {}\n'.format(auc))
        fout.write('threshold: {}\n'.format(threshold))
        # fout.write('fpr: {}\n'.format(fpr))
        # fout.write('tpr: {}\n'.format(tpr))
        fout.write('best accuracy: {}\n\n'.format(str(max(accs))))

# Loop to run test 5 times
for model in MODELS:
    print("#"*50)
    print(f"Running tests for model: {model}")
    for i in range(0, 10):
        print(f"Running test {i}")
        run_test(model, i)


