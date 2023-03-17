import os
import pickle
import numpy as np
import torch
# from tqdm import tqdm
from scipy.spatial.distance import cosine

# ========================================================

_datasetFeaturesFiles2 = {"MSD": "./checkpoints/MSD/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "Semantics_MSD": "./checkpoints/MSD/WideResNet28_10_S2M2_R/last/Fsem.mat",
                          "DAGM": "./checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "Semantics_DAGM": "./checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Fsem.mat",
                          "KTD": "./checkpoints/KTD/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "Semantics_KTD": "./checkpoints/KTD/WideResNet28_10_S2M2_R/last/Fsem.mat",
                          "KTH": "./checkpoints/KTH/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "Semantics_KTH": "./checkpoints/KTH/WideResNet28_10_S2M2_R/last/Fsem.mat",
"DTD": "./feature/DTD/CLIP.mat",
"Semantics_DTD": "./feature/DTD/gan.mat",
"EuroSAT": "./feature/EuroSAT/CLIP.mat",
"Semantics_EuroSAT": "./feature/EuroSAT/gan.mat",
"GTSRB": "./feature/GTSRB/CLIP.mat",
"Semantics_GTSRB": "./feature/GTSRB/gan.mat",
"MED-3": "./feature/MED-3/CLIP.mat",
"Semantics_MED-3": "./feature/MED-3/gan.mat",
"MT-CF": "./feature/MT-CF/CLIP.mat",
"Semantics_MT-CF": "./feature/MT-CF/gan.mat",
"RESISC45": "./feature/RESISC45/CLIP.mat",
"Semantics_RESISC45": "./feature/RESISC45/gan.mat",

                          }

_cacheDir = "./cache"
# _maxRuns = 10000
_maxRuns = 1
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None

import scipy.io as io
def _load_pickle2(file):
    data = io.loadmat(file)
    data['labels'] = data['labels'].reshape(data['labels'].shape[1])
    data['labels'] = torch.from_numpy(data['labels'])
    data['features'] = torch.from_numpy(data['features'])
    dataset = {'data': data['features'], 'labels': data['labels']}
    return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None
semantics = None #data

def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles2:
        raise NameError('Unknwown MSD: {}'.format(dsname))

    global dsName, data, semantics, labels, _randStates, _rsCfg, _min_examples, _min_examples1
    dsName = dsname
    _randStates = None
    _rsCfg = None

    if(dsname == "MSD"):
        dssem = "Semantics_MSD"
    if (dsname == "DAGM"):
        dssem = "Semantics_DAGM"
    if (dsname == "KTD"):
        dssem = "Semantics_KTD"
    if (dsname == "KTH"):
        dssem = "Semantics_KTH"
    if (dsname == "DTD"):
        dssem = "Semantics_DTD"
    if (dsname == "EuroSAT"):
        dssem = "Semantics_EuroSAT"
    if (dsname == "GTSRB"):
        dssem = "Semantics_GTSRB"
    if (dsname == "MED-3"):
        dssem = "Semantics_MED-3"
    if (dsname == "MT-CF"):
        dssem = "Semantics_MT-CF"
    if (dsname == "RESISC45"):
        dssem = "Semantics_RESISC45"

    # Loading data from files on computer
    # home = expanduser("~")

    dataset = _load_pickle2(_datasetFeaturesFiles2[dsname]) #此时dataset与output相同

    Semset = _load_pickle2(_datasetFeaturesFiles2[dssem])

    # 取每类中最小图片数作为类数
    # Computing the number of items per class in the MSD
    _min_examples = dataset["labels"].shape[0]

    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]

    #对语义向量进行处理
    _min_examples1 = Semset["labels"].shape[0]

    for i in range(Semset["labels"].shape[0]):
        if torch.where(Semset["labels"] == Semset["labels"][i])[0].shape[0] > 0:
            _min_examples1 = min(_min_examples1, torch.where(
                Semset["labels"] == Semset["labels"][i])[0].shape[0])

    # Generating semantics tensors
    semantics = torch.zeros((0, _min_examples1, Semset["data"].shape[1]))
    labels = Semset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(Semset["labels"] == labels[0])[0]
        semantics = torch.cat([semantics, Semset["data"][indices, :]
        [:_min_examples1].view(1, _min_examples1, -1)], dim=0)

        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]



def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, semantics, _min_examples, _min_examples1
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]#第一个run从20类中随机取5类进行分类

    classes = np.sort(classes)
    shuffle_indices = np.arange(_min_examples) #从0到_min_examples的图标
    shuffle_indices1 = np.arange(_min_examples1)
    dataset = None
    Semset = None
    SDset = None
    if generate:
        Semset = torch.zeros(
            (cfg['ways'], cfg['sem'], semantics.shape[2]))
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot'], data.shape[2]))#[5,16,640]
    for i in range(cfg['ways']):

        shuffle_indices = np.random.permutation(shuffle_indices) #打乱图片顺序
        shuffle_indices1 = np.random.permutation(shuffle_indices1)

        shuffle_indices1 = np.sort(shuffle_indices1)

        if generate:
            dataset[i] = data[classes[i], shuffle_indices,:][:cfg['shot']]

            Semset[i] = semantics[classes[i], shuffle_indices1, :][:cfg['sem']]

    # 合并语义与支持集+查询集数组
    if generate:
        SDset = torch.zeros(
            (cfg['ways'], cfg['sem']+cfg['shot'], data.shape[2]))

    for i in range(cfg['ways']):
        if generate:
            SDset[i] = torch.cat([dataset[i], Semset[i]], axis=0)
    dataset = SDset
    return dataset


def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "SemRandStates_{}_s{}".format(
        dsName, cfg['shot'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "sem": 15}

    setRandomStates(cfg)


    #(1000,5,16,640)
    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['sem']+cfg['shot'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)
    return dataset


# define a main code to test this module
if __name__ == "__main__":

    # print("Testing Task loader for Few Shot Learning")
    loadDataSet('MSD') # ******MSD*******

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)
    run10 = GenerateRun(10, cfg)
    run10 = GenerateRun(10, cfg)
    ds = GenerateRunSet(start=2, end=12, cfg=cfg)


