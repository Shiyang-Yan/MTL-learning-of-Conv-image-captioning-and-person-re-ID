# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import json
import pickle
basepath = '/media/shiyang/DATA1/reid_strong/'
import os
import glob
import random
import torch
with open('reid_data/CUHK-PEDES/caption_all.json') as fin:
    data = json.load(fin)

all_captions = {}
for caption in data:
    all_captions[caption['file_path']] = caption['captions']

def read_sentence_market(img_path):
    imgs = img_path.split('/')
    img_name = imgs[-1]
    img_names = img_name.split('_')
    ID = int(img_names[0])
    key_image = img_name
    path = 'reid_data/CUHK-PEDES/imgs/'+img_name
    if not os.path.exists(path) and ID != 0:
        key_image_list = glob.glob('reid_data/CUHK-PEDES/imgs/Market/'+ '%04d'%ID + '*.jpg')
        key_image = random.choice(key_image_list).replace('reid_data/CUHK-PEDES/imgs/','')
    if ID!=0:
        sentence = all_captions[key_image]
        num_captions = random.randint(0,1)
        single_sentence = sentence[num_captions]
        list_sent = single_sentence.strip().split()
        worddict_tmp = pickle.load(open('reid_data/wordlist_reid.p', 'rb'))
        wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
        wordlist_final = ['EOS'] + sorted(wordlist)
    word_vectors_all = torch.LongTensor(20).zero_()
    if ID!=0:
        for i, word in enumerate(list_sent):
            if i >= 20:
                break
            if (word not in wordlist_final):
                word = 'UNK'
            word_vectors_all[i] = wordlist_final.index(word)
    return word_vectors_all



def read_sentence_duke(img_path):
    imgs = img_path.split('/')
    all_captions = json.load(open('reid_data/dukemtmc-reid/out_unt.json', 'r'))  
    #key_image = 'duke_'+ imgs[-1]
    key_image = imgs[-1]
    #print(key_image)
    sentence = all_captions[key_image]
    sentence = sentence.replace('caption:', '')
    list_sent = sentence.strip().split()
    worddict_tmp = pickle.load(open('reid_data/wordlist_reid.p', 'rb'))
    wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
    wordlist_final = ['EOS'] + sorted(wordlist)
    #print(len(wordlist_final))
    word_vectors_all = torch.LongTensor(20).zero_()
    for i, word in enumerate(list_sent):
        if i >= 20:
            break
        if (word not in wordlist_final):
            word = 'UNK'
        word_vectors_all[i] = wordlist_final.index(word)
    return word_vectors_all

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        caption = read_sentence_duke(img_path)
        if self.transform is not None:
            img = self.transform(img)    
        return img, caption, pid, camid,  img_path
