# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import math

import torch
from torch import nn
import pickle
from torch.nn import functional as F
#basepath = '/media/shiyang/DATA1/reid_strong/reid-strong-baseline/'
worddict_tmp = pickle.load(open('reid_data/wordlist_reid.p', 'rb'))
wordlist = [l for l in iter(worddict_tmp.keys()) if l != '</S>']
wordlist_final = ['EOS'] + sorted(wordlist)
num_class = len(wordlist_final)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        #self.f_linear = nn.Linear(d_model, 256)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
           p_attn = dropout(p_attn)
        out = torch.matmul(p_attn, value).sum(2)
        return out, p_attn


    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        #out = F.relu(self.f_linear(F.relu(out))) 
        return self.linears[-1](x)


class convcap(nn.Module):

  def __init__(self, num_wordclass, num_layers=1, is_attention=True, nfeats=512, dropout=.1):
    super(convcap, self).__init__()
    self.nimgfeats = 512
    self.is_attention = is_attention
    self.nfeats = nfeats
    self.dropout = dropout

    self.emb_0 = Embedding(num_wordclass, nfeats, padding_idx=0)
    self.emb_1 = Linear(nfeats, nfeats, dropout=dropout)

    #self.imgproj = Linear(self.nimgfeats, self.nfeats, dropout=dropout)
    self.resproj = Linear(nfeats*2, self.nfeats, dropout=dropout)

    n_in = 2*self.nfeats
    n_out = self.nfeats
    self.n_layers = num_layers
    self.convs = nn.ModuleList()
    self.attention = nn.ModuleList()
    self.kernel_size = 5
    self.pad = self.kernel_size - 1
    for i in range(self.n_layers):
      self.convs.append(Conv1d(n_in, 2*n_out, self.kernel_size, self.pad, dropout))
      if(self.is_attention):
        self.attention.append(AttentionLayer(n_out, nfeats))
      n_in = n_out

    self.classifier_0 = Linear(self.nfeats, (nfeats // 2))
    self.classifier_1 = Linear((nfeats // 2), num_wordclass, dropout=dropout)
    self.bn = nn.BatchNorm1d(512)
  def forward(self, imgsfeats, imgsfc7, wordclass):

    attn_buffer = None
    wordemb = self.emb_0(wordclass)
    wordemb = self.emb_1(wordemb)
    x = wordemb.transpose(2, 1)
    batchsize, wordembdim, maxtokens = x.size()

    #y = F.relu(self.imgproj(imgsfc7))
    y = imgsfc7
    y = y.unsqueeze(2).expand(batchsize, self.nfeats, maxtokens)
    x = torch.cat([x, y], 1)

    for i, conv in enumerate(self.convs):

      if(i == 0):
        x = x.transpose(2, 1)
        residual = self.resproj(x)
        residual = residual.transpose(2, 1)
        x = x.transpose(2, 1)
      else:
        residual = x

      x = F.dropout(x, p=self.dropout, training=self.training)

      x = conv(x)
      x = x[:,:,:-self.pad]

      x = F.glu(x, dim=1)

      if(self.is_attention):
        attn = self.attention[i]
        x = x.transpose(2, 1)
        x, attn_buffer = attn(x, wordemb, imgsfeats)
        x = x.transpose(2, 1)

      x = (x+residual)*math.sqrt(.5)

    x = x.transpose(2, 1)
    multimodal_feature = (x + y.transpose(2,1))
    multimodal_feature = self.bn(multimodal_feature.transpose(2,1))
    multimodal_feature = multimodal_feature.transpose(2,1)
    x = self.classifier_0(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = self.classifier_1(x)

    x = x.transpose(2, 1)

    return x, multimodal_feature, attn_buffer



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.att = MultiHeadedAttention(8, 2048)
        self.capmodel = convcap(num_class, num_layers = 3)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, language):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        v = self.global_avgpool(x)
        x, multimodal_feature, attmap = self.capmodel(x, v, language)
        feats_out = self.att(multimodal_feature, multimodal_feature, multimodal_feature)
        feats_out = feats_out.squeeze()
        feats_all = torch.cat([v, feats_out], 1)
        return feats_all

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

