import os
import pickle
import numpy as np
import torch
# from tqdm import tqdm
from scipy.spatial.distance import cosine
import filter_sem


def countCos(v1, v2):
    s1 = np.zeros((100,))
    for i in range(0, 100):
        v3 = v2[i, :]
        s1[i] = cosine(v1, v3) #越小相似度越高
    ordered = sorted(range(len(s1)), key=lambda k: s1[k])
    return ordered

# ========================================================
# Usefull paths

_datasetFeaturesFiles2 = {"MSD": "./checkpoints/MSD/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "DAGM": "./checkpoints/DAGM/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "KTD": "./checkpoints/KTD/WideResNet28_10_S2M2_R/last/Wrn.mat",
                          "KTH": "./checkpoints/KTH/WideResNet28_10_S2M2_R/last/Wrn.mat",
"DTD": "./feature/DTD/CLIP.mat",
"EuroSAT": "./feature/EuroSAT/CLIP.mat",
"GTSRB": "./feature/GTSRB/CLIP.mat",
"MED-3": "./feature/MED-3/CLIP.mat",
"MT-CF": "./feature/MT-CF/CLIP.mat",
"RESISC45": "./feature/RESISC45/CLIP.mat",

                          }

_cacheDir = "./cache"
# _maxRuns = 10000
_maxRuns = 1000
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
sem = None

def getSem(cfg):
    return sem

def loadDataSet(dsname, cfg):
    if dsname not in _datasetFeaturesFiles2:
        raise NameError('Unknwown MSD: {}'.format(dsname))

    global dsName, data, semantics, labels, _randStates, _rsCfg, _min_examples, _min_examples1, sem
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    print("name_________________", _datasetFeaturesFiles2[dsname])
    dataset = _load_pickle2(_datasetFeaturesFiles2[dsname]) #此时dataset与output相同
    if (dsname == "MSD"):
        # ways = 20
        ways = 15
    if (dsname == "DTD"):
        ways = 11
    if (dsname == "EuroSAT"):
        ways = 3
    if (dsname == "GTSRB"):
        ways = 10
    if (dsname == "MED-3"):
        ways = 4
    if (dsname == "MT-CF"):
        ways = 3
    if (dsname == "RESISC45"):
        ways = 12
    Semset = filter_sem.dealSem(cfg['shot'], dsname, ways)
    # Semset = filter_sem.dealSem(cfg['shot'], dsname, 5)
    print("_____________________________>>>>>>>>", dataset["data"].shape[0])
    # Computing the number of items per class in the MSD
    _min_examples = dataset["labels"].shape[0]
    print("_min_exaples",_min_examples)
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples)) #取得每类中图片最小值
    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]), data.shape)


    _min_examples1 = Semset["labels"].shape[0]
    print("_min_exaples", _min_examples1)
    for i in range(Semset["labels"].shape[0]):
        if torch.where(Semset["labels"] == Semset["labels"][i])[0].shape[0] > 0:
            _min_examples1 = min(_min_examples1, torch.where(
                Semset["labels"] == Semset["labels"][i])[0].shape[0])
    print("Guaranteed number of sem per class: {:d}\n".format(_min_examples1))  # 取得每类中图片最小值
    if (cfg['shot'] == 0):
        # sem = 4
        sem = 6
    else:
        if(_min_examples1 > 50):
            _min_examples1 = _min_examples1-50
        if (_min_examples1 > 80):
            _min_examples1 = _min_examples1 - 80
        if (_min_examples1 > 20):
            _min_examples1 = _min_examples1 - 20
        if (_min_examples1 > 10):
            _min_examples1 = _min_examples1 - 10
        sem = _min_examples1
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

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)
    shuffle_indices1 = np.arange(_min_examples1)
    dataset = None
    Semset = None
    SDset = None
    if generate:
        Semset = torch.zeros(
            (cfg['ways'], sem, semantics.shape[2]))
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))#[5,16,640]
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        shuffle_indices1 = np.random.permutation(shuffle_indices1)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices, :][:cfg['shot']+cfg['queries']]
            Semset[i] = semantics[classes[i], shuffle_indices1, :][:sem]

    if generate:
        SDset = torch.zeros(
            (cfg['ways'], sem+cfg['shot']+cfg['queries'], data.shape[2]))

    for i in range(cfg['ways']):
        if generate:

            SDset[i] = torch.cat([Semset[i], dataset[i]], axis=0)
    dataset = SDset

    # import SelfAttention
    # C = 640
    # num_head = 8
    # MHSA = SelfAttention.MultiHead_SelfAttention(C, num_head)
    # dataset = MHSA(dataset)
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

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways'],))
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
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    #(1000,5,16,640)
    dataset = torch.zeros(
        (end-start, cfg['ways'], sem+cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)
    return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('MSD') # ******MSD*******

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)
    run10 = GenerateRun(10, cfg)
    run10 = GenerateRun(10, cfg)
    ds = GenerateRunSet(start=2, end=12, cfg=cfg)


