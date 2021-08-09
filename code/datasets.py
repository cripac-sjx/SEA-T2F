from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
import torch.nn.functional as Func
import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import pdb
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, cap_lens, keys, wrong_caps, w_cap_lens, attr = data

    if cfg.CUDA:
        for i in range(len(imgs)):
            captions[i] = Variable(captions[i]).cuda()
            cap_lens[i] = Variable(cap_lens[i]).cuda()

            wrong_caps[i] = Variable(wrong_caps[i]).cuda()
            w_cap_lens[i] = Variable(w_cap_lens[i]).cuda()

    return [imgs, captions, cap_lens,
            keys, wrong_caps, w_cap_lens, attr]
def get_imgs(img_path, normalize):

    img = Image.open(img_path).convert('RGB')
    img=normalize(img)
    img=img.view(1,3,img.shape[1],img.shape[2])
    imgs=[]
    img1=Func.interpolate(img,scale_factor=1/16)
    img1=img1.view(3,img1.shape[2],img1.shape[3])
    imgs.append(img1)
    img2=Func.interpolate(img,scale_factor=1/8)
    img2=img2.view(3,img2.shape[2],img2.shape[3])
    imgs.append(img2)
    img3=Func.interpolate(img,scale_factor=1/4)
    img3=img3.view(3,img3.shape[2],img3.shape[3])
    imgs.append(img3)
    return imgs


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',
                 attr_path='../data/CelebAText-HQ/CelebAMask-HQ-attribute-anno.txt', selected_attrs=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.split=split

        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.attr2idx = {}
        self.idx2attr = {}

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir

        self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words, self.attr = self.load_text_data(data_dir, split,'captions.pickle')


        self.number_example = len(self.filenames)

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%scaptions/%s.txt' % (data_dir, filenames[i].split('.')[0])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]


    def load_text_data(self, data_dir, split,pickles):
        filepath = os.path.join(data_dir, pickles)
        train_name = data_dir + 'data_flist/' + 'train' + '.flist'
        test_name = data_dir + 'data_flist/' + 'test' + '.flist'


        train_names = self.load_filenames(train_name)
        test_names = self.load_filenames(test_name)

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        #loda attributes
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]

        img_name=[]
        labels=[]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')
            labels.append(label)
            img_name.append(filename)
        attr = dict(map(lambda x,y:[x,y], img_name, labels))
        return filenames, captions, ixtoword, wordtoix, n_words, attr

    def load_filenames(self, split_name):
        f = open(split_name, 'r')
        filenames = f.readlines()
        file_name = []
        for file in filenames:
            item = file.split('/')[3].split('\n')[0]
            file_name.append(item)
        return file_name

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len
    def get_label(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.labels[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):

        key = self.filenames[index]

        bbox = None
        if self.split=='valid':
            img_name='../valid/%s'%(key.split('.')[0]+'.png')
            img = Image.open(img_name).convert('RGB')
            imgs=self.norm(img)
        else:
            img_name = '%sCelebAimg/%s' %(self.data_dir, key)
            imgs = get_imgs(img_name, normalize=self.norm)

        # select 10 sentence
        caps=[]
        wrong_caps=[]
        cap_lens=[]
        wrong_cap_lens=[]
        for ix in range(self.embeddings_num):
            new_sent_ix = index * self.embeddings_num + ix
            cap, cap_len = self.get_caption(new_sent_ix)
            caps.append(cap)
            cap_lens.append(cap_len)
            # randomly select a mismatch sentence
            wrong_idx = random.randint(0, len(self.filenames))
            wrong_new_sent_ix = wrong_idx * self.embeddings_num + ix
            wrong_cap, wrong_cap_len = self.get_caption(wrong_new_sent_ix)
            wrong_caps.append(wrong_cap)
            wrong_cap_lens.append(wrong_cap_len)

        if self.split=='valid':
            flag=img_name.split('/')[-1].split('.')[0]+'.jpg'
        else:
            flag=img_name.split('/')[-1]
        attr=self.attr[flag]
        attr=np.array(attr,dtype=int)
        attr=attr.reshape(attr.shape[0],1)

        return imgs, caps, cap_lens, key, wrong_caps, wrong_cap_lens, attr

    def __len__(self):
        return len(self.filenames)