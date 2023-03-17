import os
import pickle
import numpy as np
import torch
# from tqdm import tqdm

# ========================================================
# Usefull paths
_datasetFeaturesFiles = {"Aluminum": "./checkpoints/Aluminum/WideResNet28_10_S2M2_R/last/output2.plk",
                         "MSD": "./checkpoints/MSD/WideResNet28_10_S2M2_R/last/output2.plk"}
_cacheDir = "./cache"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        # print('222222222222222data22222222222222222', data)
        dataset = dict()
        # print('222222222222222dataset22222222222222222', Aluminum)
        # Aluminum['data'] = torch.FloatTensor(np.stack(data, axis=0))
        # Aluminum['labels'] = torch.LongTensor(np.concatenate(labels))
        dataset['data'] = np.stack(data, axis=0)
        dataset['labels'] = np.concatenate(labels)
        return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown Aluminum: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    print(dsname)
    print("name_________________", _datasetFeaturesFiles[dsname])
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])
    print("_____________________________>>>>>>>>", dataset["data"].shape[0])
    print("222222222222222222从output打印出来的dataset：2222222222222", dataset)
    feat_data = dataset['data']  # image data
    labels = dataset['labels'].astype(int)  # class labels
    print('2222222222222222222feat_data:', feat_data)
    print('2222222222222222222labels:', labels)

    # Computing the number of items per class in the Aluminum
    _min_examples = dataset["labels"].shape[0]
    print("_min_exaples",_min_examples)
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        print('data:',data)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]), data.shape)


def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    # print("classes:", classes)
    # print("classes:", classes)
    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    print("classes:",classes)
    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        print("i:",i)
        print("range(cfg['ways']):", range(cfg['ways']))
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,:][:cfg['shot']+cfg['queries']]
            print("Aluminum[i]:",dataset[i])
    # print("________dataset", Aluminum.shape)
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
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
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

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    # print(Aluminum)
    print(dataset.shape)
    for iRun in range(end-start):
        print('9999999999999999999999999999999999999999999')
        print(start+iRun)
        print(start)
        print(iRun)
        print(cfg)
        dataset[iRun] = GenerateRun(start+iRun, cfg)
        print('8888888888888889999999999999999999999999999999999999999999')
    return dataset


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('Aluminum') # ******Aluminum*******

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)
    run10 = GenerateRun(10, cfg)
    run10 = GenerateRun(10, cfg)
    ds = GenerateRunSet(start=2, end=12, cfg=cfg)


