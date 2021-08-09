import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query) #multiply
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn) 

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)

    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class SpatialAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(SpatialAttention, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.conv_context2 = conv1x1(cdf, 64*64)
        self.conv_context3 = conv1x1(cdf, 128*128)
        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        # make the softmax on the dimension 1
        attn = self.sm(attn)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        weightedContext = torch.bmm(sourceT, attn)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn
class MultiAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(MultiAttention, self).__init__()
        self.conv_concat=conv1x1(10*idf, idf)
        self.conv_b=conv1x1(1,idf)
        self.conv_context = conv1x1(cdf, idf)
        self.ReLU=nn.ReLU()
        self.LN = nn.LayerNorm(idf, eps=0, elementwise_affine=True)
        self.sm = nn.Softmax()
        self.mask = None
        self.idf = idf

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def spatial_attn(self, targetT, context, batch_size):
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        weight=torch.bmm(targetT,sourceT) #batch*wh*idf batch *idf*T
        weight = F.softmax(weight, dim=-1)#batch*wh*T
        sourceT = torch.transpose(sourceT,1,2).contiguous()
        attn = torch.matmul(weight, sourceT)

        return attn


    def forward(self, input, contexts,head=8):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        batch_size, ih, iw =input.size(0), input.size(2), input.size(3)
        queryL = ih * iw

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous() #Q:batch*M*d batch*4096*32

        for i in range(len(contexts)):
            context=contexts[i]
            attn=self.spatial_attn(targetT, context, batch_size)
            if i==0:
                Attn_concat = attn
            else:
                Attn_concat=torch.cat((Attn_concat,attn),-1)

        IM=torch.ones(queryL).repeat(batch_size,1).unsqueeze(2).unsqueeze(3).cuda()
        Attn_concat=Attn_concat.unsqueeze(3)
        x_attn=self.conv_concat(Attn_concat.transpose(1,2)).squeeze(3).transpose(1,2)
        x_attn=x_attn+self.conv_b(IM.transpose(1,2)).transpose(1,2).squeeze(3)
        x_attn=self.ReLU(x_attn)
        x_attn=x_attn+targetT
        x_attn=self.LN(x_attn)

        return x_attn