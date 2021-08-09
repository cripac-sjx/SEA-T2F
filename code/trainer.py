from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import scipy
from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET, Attr_Discriminator, network_9layers,network_29layers,LightCNN_29Layers_v2

from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import discriminator_loss, generator_loss, KL_loss

import os
import time
import numpy as np
import sys
import pdb
import torchvision.utils as vutils
# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, n_words, ixtoword):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.n_words = n_words
        self.n_labels=41
        self.ixtoword = ixtoword
        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)
        self.l1_loss = nn.L1Loss()

    def build_models(self):

        if cfg.TRAIN.NET_E == '':
            print('Error: no pretrained text-image encoders')
            return

        image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
        img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = \
            torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
        image_encoder.load_state_dict(state_dict)
        for p in image_encoder.parameters():
            p.requires_grad = False
        print('Load image encoder from:', img_encoder_path)
        image_encoder.eval()

        text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E,
                       map_location=lambda storage, loc: storage)
        text_encoder.load_state_dict(state_dict)
        for p in text_encoder.parameters():
            p.requires_grad = False
        print('Load text encoder from:', cfg.TRAIN.NET_E)
        text_encoder.eval()

        netsD = []
        if cfg.GAN.B_DCGAN:
            if cfg.TREE.BRANCH_NUM ==1:
                from model import D_NET64 as D_NET
            elif cfg.TREE.BRANCH_NUM == 2:
                from model import D_NET128 as D_NET
            else:  # cfg.TREE.BRANCH_NUM == 3:
                from model import D_NET256 as D_NET
            netG = G_DCGAN()
            netsD = [D_NET(b_jcu=False)]
        else:
            from model import D_NET64, D_NET128, D_NET256
            netG = G_NET()
            if cfg.TREE.BRANCH_NUM > 0:
                netsD.append(D_NET64())
            if cfg.TREE.BRANCH_NUM > 1:
                netsD.append(D_NET128())
            if cfg.TREE.BRANCH_NUM > 2:
                netsD.append(D_NET256())
        netG.apply(weights_init)

        
        for i in range(len(netsD)):
            netsD[i].apply(weights_init)
        print('# of netsD', len(netsD))
        #
        epoch = 0
        if cfg.TRAIN.NET_G != '':
            state_dict = \
                torch.load(cfg.TRAIN.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            if cfg.TRAIN.B_NET_D:
                Gname = cfg.TRAIN.NET_G
                for i in range(len(netsD)):
                    s_tmp = Gname[:Gname.rfind('/')]
                    Dname = '%s/netD%d.pth' % (s_tmp, i)
                    print('Load D from: ', Dname)
                    state_dict = \
                        torch.load(Dname, map_location=lambda storage, loc: storage)
                    netsD[i].load_state_dict(state_dict)

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            netG.cuda()
            # style_loss = style_loss.cuda()
            for i in range(len(netsD)):
                netsD[i].cuda()

        attr_D=Attr_Discriminator()

        return [text_encoder, image_encoder, netG, netsD, epoch, attr_D]


    def define_optimizers(self, netG, netsD, attrD):
        optimizersD = []
        num_Ds = len(netsD)
        for i in range(num_Ds):
            opt = optim.Adam(netsD[i].parameters(),
                             lr=cfg.TRAIN.DISCRIMINATOR_LR,
                             betas=(0.5, 0.999))
            optimizersD.append(opt)

        optimizerG = optim.Adam(netG.parameters(),
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        optimizers_att = optim.Adam(attrD.parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        return optimizerG, optimizersD, optimizers_att

    def prepare_labels(self):
        batch_size = self.batch_size
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(batch_size)))
        if cfg.CUDA:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def save_model(self, netG, avg_param_G, netsD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(),
            '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        #
        for i in range(len(netsD)):
            netD = netsD[i]
            torch.save(netD.state_dict(),
                '%s/netD%d.pth' % (self.model_dir, i))
        print('Save G/Ds models.')

    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires


    def train(self):

        text_encoder, image_encoder, netG, netsD, start_epoch, attr_D = self.build_models()
        avg_param_G = copy_G_params(netG)

        optimizerG, optimizersD, optimizers_att = self.define_optimizers(netG, netsD, attr_D)
        real_labels, fake_labels, match_labels = self.prepare_labels()

        batch_size = self.batch_size
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            attr_D=attr_D.cuda()

        gen_iterations = 0

        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            data_iter = iter(self.data_loader)
            step = 0
            num_D=3
            while step < self.num_batches:

                data = data_iter.next()
                imgs, captions, cap_lens, keys, wrong_caps, \
                            wrong_caps_len, attr= data
                class_ids=None

                hidden = text_encoder.init_hidden(batch_size)
                words_emb=[]
                sent_emb=[]
                w_words_emb=[]
                w_sent_emb=[]
                for i in range(len(captions)):
                    sort_cap_lens, sort_cap_index = torch.sort(cap_lens[i], 0, True)
                    captions[i]=captions[i][sort_cap_index].squeeze().cuda()
                    wd_embs, s_emb = text_encoder(captions[i], sort_cap_lens, hidden)

                    wd_embs[sort_cap_index]=wd_embs
                    s_emb[sort_cap_index] = s_emb
                    wd_embs, s_emb = wd_embs.detach(), s_emb.detach()
                    words_emb.append(wd_embs)
                    sent_emb.append(s_emb)

                # wrong word and sentence embeddings
                    w_sort_cap_lens, w_sort_cap_index = torch.sort(wrong_caps_len[i], 0, True)
                    wrong_caps[i] = wrong_caps[i][w_sort_cap_index].squeeze().cuda()
                    w_wd_embs, w_s_emb = text_encoder(wrong_caps[i], w_sort_cap_lens, hidden)
                    w_wd_embs[w_sort_cap_index] = w_wd_embs
                    w_s_emb[w_sort_cap_index] = w_s_emb
                    w_wd_embs, w_s_emb = w_wd_embs.detach(), w_s_emb.detach()
                    w_words_emb.append(w_wd_embs)
                    w_sent_emb.append(w_s_emb)
                for i in range(len(words_emb)):
                    if i==0:
                        words_embs=words_emb[i]
                        w_words_embs=w_words_emb[i]
                    else:
                        words_embs=torch.cat((words_embs,words_emb[i]),2)
                        w_words_embs=torch.cat((w_words_embs,w_words_emb[i]),2)
                noise.data.normal_(0, 1)


                mask = None

                fake_imgs, mu, logvar = netG(num_D, noise, sent_emb,  words_emb, mask)

                errD_total = 0
                D_logs = ''

                attr_D.zero_grad()
                att_cls = attr_D(imgs[2].cuda())
                attr_labels = attr.view(att_cls.shape).float()
                err_att = 0.01 * self.classification_loss(att_cls.cpu(), attr_labels.cpu())
                err_att.backward()
                optimizers_att.step()
                if step %  5==0:
                    for i in range(num_D):
                        netsD[i].zero_grad()
                        errD = discriminator_loss(netsD[i], imgs[i].cuda(), fake_imgs[i],
                                                  sent_emb, real_labels.cuda(), fake_labels.cuda())

                        # backward and update parameters
                        errD.backward(retain_graph=True)
                        optimizersD[i].step()
                        errD_total += errD
                        D_logs += 'errD%d: %.2f ' % (i, errD)

                #==================================================================================#
                                              #train the generator
                #==================================================================================#

                step += 1
                gen_iterations += 1

                netG.zero_grad()
                #compute attribute classification loss
                if num_D==3:
                    fake_images=torch.tensor(fake_imgs[2])
                    att_cls = attr_D(fake_images)
                    attr_labels =  attr.view(att_cls.shape).float()
                    err_att = self.classification_loss(att_cls.cpu(), attr_labels.cpu())

                errG_total, G_logs = \
                    generator_loss(num_D, netsD, image_encoder, fake_imgs, real_labels,
                                   words_embs, sent_emb, match_labels, cap_lens, class_ids)
                kl_loss=0
                for i in range(len(mu)):
                    kl_loss += KL_loss(mu[i], logvar[i])
                errG_total += kl_loss
                if num_D==3:
                    errG_total +=0.02*err_att.cuda()
                G_logs += 'kl_loss: %.2f ' % kl_loss
                G_logs+='cls_loss: %.2f' % err_att
                errG_total.backward()
                optimizerG.step()
                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 100 == 0:
                    print(D_logs + '\n' + G_logs)

            end_t = time.time()

            print('''[%d/%d][%d]
                  Loss_D: %.2f Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total, errG_total,
                     end_t - start_t))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                self.save_model(netG, avg_param_G, netsD, epoch)
                vutils.save_image(fake_imgs[2].data,
                                  '%s/%s/fake_samples_epoch_%03d.png' % (
                                  self.model_dir.split('/Model')[0], 'Image', epoch),
                                  normalize=True)

        self.save_model(netG, avg_param_G, netsD, self.max_epoch)

    def save_singleimages(self, images, filenames, save_dir,
                          split_dir, sentenceID=0):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d.jpg' % (s_tmp, sentenceID)
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def sampling(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            if split_dir == 'test':
                split_dir = 'valid'
            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            netG.apply(weights_init)
            netG.cuda()
            netG.eval()



            text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
            img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
            state_dict = \
                torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
            image_encoder.load_state_dict(state_dict)
            for p in image_encoder.parameters():
                p.requires_grad = False
            print('Load image encoder from:', img_encoder_path)
            image_encoder = image_encoder.cuda()
            image_encoder.eval()

            batch_size = self.batch_size
            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
            noise = noise.cuda()

            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)

            # the path to save generated images
            s_tmp = model_dir[:model_dir.rfind('.pth')]
            save_dir = '%s/%s_5' % (s_tmp, split_dir)
            mkdir_p(save_dir)

            cnt = 0
            idx = 0 ###
            for _ in range(1):
                for step, data in enumerate(self.data_loader, 0):
                    cnt += batch_size
                    if step % 100 == 0:
                        print('step: ', step)

                    imgs, captions, cap_lens, keys, wrong_caps, \
                    wrong_caps_len, attr = data

                    hidden = text_encoder.init_hidden(batch_size)
                    words_emb = []
                    sent_emb = []
                    for i in range(len(captions)):
                        sort_cap_lens, sort_cap_index = torch.sort(cap_lens[i], 0, True)
                        captions[i] = captions[i][sort_cap_index].squeeze().cuda()
                        wd_embs, s_emb = text_encoder(captions[i], sort_cap_lens, hidden)
                        wd_embs[sort_cap_index] = wd_embs
                        s_emb[sort_cap_index] = s_emb
                        wd_embs, s_emb = wd_embs.detach(), s_emb.detach()
                        words_emb.append(wd_embs)
                        sent_emb.append(s_emb)

                    mask = None
                    noise.data.normal_(0, 1)
                    #h_code, fake_img1 = netG0(noise, sent_emb)
                    num_D=3
                    fake_imgs, _, _ = netG(num_D, noise, sent_emb, words_emb, mask)
                    f_imgs=F.interpolate(fake_imgs[2],scale_factor=0.5)


                    for j in range(batch_size):
                        s_tmp = '%s/' % (save_dir)
                        folder = s_tmp[:s_tmp.rfind('/')]
                        if not os.path.isdir(folder):
                            print('Make a new folder: ', folder)
                            mkdir_p(folder)
                        k = -1
                        im = fake_imgs[k][j].data.cpu().numpy()
                       # im = (im + 1.0) * 127.5
                        im = 255 * (im - np.min(im)) / (np.max(im) - np.min(im))
                        im = im.astype(np.uint8)
                        im = np.transpose(im, (1, 2, 0))
                        im = Image.fromarray(im)
                        fullpath='%s%s.png'%(s_tmp,keys[j].split('.')[0])
                        idx = idx+1
                        im.save(fullpath)


    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for models is not found!')
        else:
            text_encoder = \
                RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
            state_dict = \
                torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
            text_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', cfg.TRAIN.NET_E)
            text_encoder = text_encoder.cuda()
            text_encoder.eval()

            checkpoint_path = "../models/netG_055.pth"
            netG0 = NetG(cfg.GAN.GF_DIM, 100)
            netG0.load_state_dict(torch.load(checkpoint_path))
            netG0 = netG0.cuda()
            netG0.eval()

            if cfg.GAN.B_DCGAN:
                netG = G_DCGAN()
            else:
                netG = G_NET()
            s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
            model_dir = cfg.TRAIN.NET_G
            state_dict = \
                torch.load(model_dir, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', model_dir)
            netG.cuda()
            netG.eval()
            for key in data_dic:
                save_dir = '%s/%s' % (s_tmp, key)
                mkdir_p(save_dir)
                captions, cap_lens, sorted_indices = data_dic[key]

                batch_size=1
                nz = cfg.GAN.Z_DIM
                captions = Variable(torch.from_numpy(captions), volatile=True)
                cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)

                captions = captions.cuda()
                cap_lens = cap_lens.cuda()
                for i in range(1): 
                    noise = Variable(torch.FloatTensor(batch_size, nz), volatile=True)
                    noise = noise.cuda()

                    hidden = text_encoder.init_hidden(10)

                    sort_cap_lens, sort_cap_index = torch.sort(cap_lens, 0, True)
                    captions = captions[sort_cap_index].squeeze().cuda()
                    wd_embs, s_emb = text_encoder(captions, sort_cap_lens, hidden)
                    wd_embs[sort_cap_index] = wd_embs
                    s_emb[sort_cap_index] = s_emb
                    wd_embs, s_emb = wd_embs.detach(), s_emb.detach()

                    words_emb=wd_embs
                    sent_emb=s_emb

                    for i in range(len(words_emb)):
                        if i == 0:
                            words_embs = words_emb[i]
                        else:
                            words_embs = torch.cat((words_embs, words_emb[i]), 1)
                    #words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)

                    mask=None

                    noise.data.normal_(0, 1)
                    #h_code, fake_img1 = netG0(noise, sent_emb)
                    num_D = 3
                    fake_imgs, attention_maps, _, _ = netG(num_D, noise, sent_emb.view(sent_emb.shape[0],1,sent_emb.shape[1]),
                                                      words_embs.view(1,words_embs.shape[0],words_embs.shape[1]), mask)

                    cap_lens_np = cap_lens.cpu().data.numpy()
                    for j in range(batch_size):
                        save_name = '%s/%d_s_%d' % (save_dir, i, sorted_indices[j])
                        for k in range(len(fake_imgs)):
                            im = fake_imgs[k][j].data.cpu().numpy()
                            im = (im + 1.0) * 127.5
                            im = im.astype(np.uint8)
                            im = np.transpose(im, (1, 2, 0))
                            im = Image.fromarray(im)
                            fullpath = '%s_g%d.png' % (save_name, k)
                            im.save(fullpath)


    def classification_loss(self, logit, target):
        """Compute binary or softmax cross entropy loss."""
        return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)

    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        """Returns cosine similarity between x1 and x2, computed along dim.
        """
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
