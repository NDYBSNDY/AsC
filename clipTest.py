import os
import clip
import torch
import json
from PIL import Image
import numpy as np
import scipy.io as io
from scipy.io import *
import collections
import pickle
import math


def MSDgetclip():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    data_path = 'filelists/MSD/clip_20.json'
    f = open(data_path, 'r', encoding='utf-8')
    data = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
    class_name = data['label_names']
    img_path = data['image_names']
    img_id = np.array(data['image_labels'])
    class_id = np.arange(0, 20, 1)

    # Prepare the inputs
    image = Image.open(img_path[0])
    images_input = preprocess(image).unsqueeze(0).to(device)
    for i in range(1, 872):
        image = Image.open(img_path[i])
        image_input = preprocess(image).unsqueeze(0).to(device)
        images_input = torch.cat((images_input, image_input), 0)
    # class_id = img_id[1]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(images_input)
        text_features = model.encode_text(text_inputs)

  
    image_features = np.array(image_features)
    text_features = np.array(text_features)
    savemat("checkpoints/MSD/WideResNet28_10_S2M2_R/last/Fsem.mat",
            {'features': text_features, 'labels': class_id})
    savemat("checkpoints/MSD/WideResNet28_10_S2M2_R/last/Wrn.mat", {'features': image_features, 'labels': img_id})

    # # Pick the top 5 most similar labels for the image
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # values, indices = similarity[0].topk(5)

    # # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{class_name[index]:>16s}: {100 * value.item():.2f}%")
    #
    # print("\nReal:" + class_name[class_id])
    # print(img_path[1])


# MSDgetclip()


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def savePlk():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    data_path = 'filelists/MSD/clip_20.json'
    f = open(data_path, 'r', encoding='utf-8')
    data = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
    class_name = data['label_names']
    img_path = data['image_names']
    img_id = np.array(data['image_labels'])
    class_id = np.arange(0, 20, 1)

    # Prepare the inputs
    image = Image.open(img_path[0])
    images_input = preprocess(image).unsqueeze(0).to(device)
    for i in range(1, 872):
        image = Image.open(img_path[i])
        image_input = preprocess(image).unsqueeze(0).to(device)
        images_input = torch.cat((images_input, image_input), 0)
    # class_id = img_id[1]
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(images_input)
        text_features = model.encode_text(text_inputs)

    output_dict = collections.defaultdict(list)
    # 生成提取结果 语义的和图片的
    # image_features = np.array(image_features)
    # text_features = np.array(text_features)
    for out, label in zip(image_features, img_id):
        output_dict[label.item()].append(out)
    for i in range(0, len(text_features)):
        output_dict[i].insert(0, text_features[i])
    save_pickle('checkpoints/MSD/WideResNet28_10_S2M2_R/last/output_clip.plk', output_dict)


# savePlk()

# # Load the model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load('ViT-B/32', device)
#
# data_path = 'filelists/MSD/clip_q15.json'
# f = open(data_path, 'r', encoding='utf-8')
# data = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
# class_name = data['label_names']
# img_path = data['image_names']
# img_id = np.array(data['image_labels'])
# class_id = np.arange(0, 20, 1)
#
# # Prepare the inputs
# image = Image.open(img_path[0])
# images_input = preprocess(image).unsqueeze(0).to(device)
# for i in range(1, 75):
#     image = Image.open(img_path[i])
#     image_input = preprocess(image).unsqueeze(0).to(device)
#     images_input = torch.cat((images_input, image_input), 0)
# # class_id = img_id[1]
# text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)
#
# # Calculate features
# with torch.no_grad():
#     image_features = model.encode_image(images_input)
#     text_features = model.encode_text(text_inputs)
#
# # Pick the top 5 most similar labels for the image
# image_features /= image_features.norm(dim=-1, keepdim=True)
# text_features /= text_features.norm(dim=-1, keepdim=True)
# similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
# values, allindices = similarity[0].topk(1)
# for i in range(1, 75):
#     values, indices = similarity[i].topk(1)
#     allindices = torch.cat((allindices, indices), 0)
#
# allindices = allindices.cuda()
# img_id = torch.from_numpy(img_id)
# img_id = img_id.cuda()
# matches = img_id.eq(allindices).float()
# # acc_test = matches[:].mean(1)
# m = matches.mean().item()
#
# # Print the result
# print("\nTop predictions:\n")
# for value, index in zip(values, indices):
#     print(f"{class_name[index]:>16s}: {100 * value.item():.2f}%")


def MSDgetWRN():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    data_path = 'filelists/MSD/clip_20_PATH.json'
    f = open(data_path, 'r', encoding='utf-8')
    data = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
    class_name = data['label_names']
    img_path = np.array(data['image_names'])
    img_id = np.array(data['image_labels'])
    class_id = np.arange(0, 20, 1)

    # Prepare the inputs
    image = Image.open(img_path[0])
    images_input = preprocess(image).unsqueeze(0).to(device)
    for i in range(1, 872):
        image = Image.open(img_path[i])
        image_input = preprocess(image).unsqueeze(0).to(device)
        images_input = torch.cat((images_input, image_input), 0)
    # class_id = img_id[1]


    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(images_input)

    # image_features /= image_features.norm(dim=-1, keepdim=True)
    
    image_features = image_features.cpu()
    image_features = np.array(image_features)
    features2 = image_features[:667]
    labels2 = img_id[:667]
    features3 = image_features[667:]
    labels3 = img_id[667:]
    savemat("GAN/MSD/Wrn_clip1.mat", {'features': image_features, 'labels': img_id})
    savemat("GAN/MSD/Wrn_clip2.mat", {'features': features2, 'labels': labels2})
    savemat("GAN/MSD/Wrn_clip3.mat", {'features': features3, 'labels': labels3})


    img_path = img_path.reshape((872, 1))
    image_features = image_features.reshape((512, 872))
    img_id = img_id.reshape((872, 1))
    # savemat("GAN/MSD/Wrn_GAN.mat", {'image_files': img_path, 'features': image_features, 'labels': img_id})



# MSDgetWRN()

def MSDgetSp():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    data_path = 'filelists/MSD/clip_20_PATH.json'
    f = open(data_path, 'r', encoding='utf-8')
    data = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
    class_name = data['label_names']
    img_path = data['image_names']
    img_id = np.array(data['image_labels'])
    class_id = np.arange(0, 20, 1)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_name]).to(device)
    class_name = np.array(class_name)
    # Calculate features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    # text_features /= text_features.norm(dim=-1, keepdim=True)

    
    text_features = text_features.cpu()
    text_features = np.array(text_features)

    featuresClip = text_features[15:]
    class_id = class_id.reshape(1, 20)
    labelsClip = class_id[:, 15:]

    # savemat("GAN/MSD/Fsem-clip.mat", {'features': featuresClip, 'labels': labelsClip})

    
    num = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 22, 28, 36, 45, 36, 45, 48, 42, 45, 25]
    # test_unseen_loc3 （18 - 20）
    start = 0
    last = 0
    for i in range(0, 15):
        last = last + num[i]
    trainval_loc = np.arange(start, last, 1)
    last = 0
    for i in range(0, 15):
        if (i < 3):
            start = start + num[i]
        last = last + num[i]
    train_loc = np.arange(start, last, 1)
    start = 0
    last = 0
    for i in range(0, 5):
        last = last + num[i]
    val_loc = np.arange(start, last, 1)
    start = 0
    last = 0
    # for i in range(0, 12):
    #     if (i < 7):
    #         start = start + num[i]
    #     last = last + num[i]
    # test_seen_loc = np.arange(start, last, 1)
    for i in range(0, 15):
        last = last + num[i]
    test_seen_loc = np.arange(start, last, 1)
    # start = 0
    # last = 0
    # for i in range(0, 20):
    #     if (i < 17):
    #         start = start + num[i]
    #     last = last + num[i]
    # test_unseen_loc = np.arange(start, last, 1)
    start = 0
    last = 0
    # for i in range(0, 20):
    #     last = last + num[i]
    # test_unseen_loc = np.arange(start, last, 1)
    for i in range(0, 20):
        if (i < 15):
            start = start + num[i]
        last = last + num[i]
    test_unseen_loc = np.arange(start, last, 1)

    trainval_loc = trainval_loc.reshape((667, 1))
    train_loc = train_loc.reshape((517, 1))
    val_loc = val_loc.reshape((250, 1))
    test_seen_loc = test_seen_loc.reshape((667, 1))
    test_unseen_loc = test_unseen_loc.reshape((205, 1))


    text_features = text_features.reshape((512, 20))
    class_name = class_name.reshape((20, 1))

    savemat("GAN/MSD/att_splits.mat",
            {'trainval_loc': trainval_loc, 'train_loc': train_loc, 'val_loc': val_loc,
                'test_seen_loc': test_seen_loc, 'test_unseen_loc': test_unseen_loc,
                'att': text_features, 'allclasses_names': class_name})

MSDgetSp()

import scipy.io as io
def MSDgetFs():
    path = 'GAN/MSD/Fsem_lossG0.0329.mat'
    data = io.loadmat(path)
    text_features = data['features']
    labels = data['labels']
    features2 = text_features[:1500]
    labels2 = labels[:, :1500]
    features3 = text_features[1500:]
    labels3 = labels[:, 1500:]

    savemat("GAN/MSD/Fsem_clip2.mat", {'features': features2, 'labels': labels2})
    savemat("GAN/MSD/Fsem_clip3.mat", {'features': features3, 'labels': labels3})

# MSDgetFs()