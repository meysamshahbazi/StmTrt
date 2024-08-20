import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
import os.path as osp
import torch
import torch.nn as nn 
from copy import deepcopy
import time
import math
from collections import OrderedDict
import torch.nn.functional as F
from utils import *

class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.

        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size
        pad: int
            padding on each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x




class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)





class AdjustLayer1(nn.Module):
    def __init__(self):
        self.default_hyper_params = dict(
        in_channels=40,
        out_channels=20,
        )
        super(AdjustLayer1, self).__init__()

    def forward(self, x):
        return self.adjustor(x)

    def update_params(self):
        # super().update_params()
        in_channels = 40 # self.default_hyper_params['in_channels']
        out_channels = 20 #self.default_hyper_params['out_channels']
        self.adjustor = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self._init_weights()

    def _init_weights(self):
        conv_weight_std = 0.01
        for m in [self.adjustor, ]:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=conv_weight_std)  # conv_weight_std=0.01


class ConvModule(nn.Module):
    def __init__(self, in_channels, mdim):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.conv_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mdim, mdim, kernel_size=3, padding=1),
        )

    def forward(self, m):
        m = self.conv1(m)
        m = self.conv_layers(m)
        return m



class STMHead1(nn.Module):
    # default_hyper_params = dict(
    #     total_stride=8,
    #     score_size=25,
    #     q_size=289, #289
    #     input_size_adapt=False,
    #     in_channels=512,
    # )

    def __init__(self, ):
        super(STMHead1, self).__init__()
        self.default_hyper_params = dict(
        total_stride=8,
        score_size=25,
        q_size=200,#
        input_size_adapt=False,
        in_channels=20,
        )
        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

    def memory_read(self, fm, fq):
        B, C, T, H, W = fm.size()
        # print(B, C, T, H, W)
        fm0 = fm.clone()
        fq0 = fq.clone()

        # fm = fm.view(B, C, T * H * W)  # B, C, THW
        fm = fm.reshape(B, C, T * H * W)
        fm = torch.transpose(fm, 1, 2)  # B, THW, C
        fq = fq.view(B, C, H * W)  # B, C, HW

        # w = torch.bmm(fm, fq) / math.sqrt(C)  # B, THW, HW
        w = torch.bmm(fm, fq) / math.sqrt(512.0)
        w = torch.softmax(w, dim=1)

        fm1 = fm0.view(B, C, T * H * W)  # B, C, THW
        mem_info = torch.bmm(fm1, w)  # (B, C, THW) x (B, THW, HW) = (B, C, HW)
        mem_info = mem_info.view(B, C, H, W)

        y = torch.cat([mem_info, fq0], dim=1)
        return y

    def solve(self, y):
        cls_feat = self.cls_ctr(y)
        reg_feat = self.reg(y)
        classification = self.cls_score(cls_feat)
        centerness = self.ctr_score(cls_feat)
        regression = self.reg_offsets(reg_feat)

        return classification, centerness, regression

    def forward(self, fm, fq, q_size=0):
        # print("fm: ",fm.shape)
        # print("fq: ",fq .shape)
        y = self.memory_read(fm, fq)
        cls_score, ctr_score, offsets = self.solve(y)
        # print(cls_score.shape,  "|",ctr_score.shape, "|",offsets.shape)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)

        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)

        offsets = torch.exp(self.si * offsets + self.bi) * self.total_stride

        # bbox decoding
        # MeySha: we comment this becuse of wed dont have input_size_adapt
        # if self.default_hyper_params["input_size_adapt"] and q_size > 0:
        #     score_offset = (q_size - 1 - (offsets.size(-1) - 1) * self.total_stride) // 2
        #     fm_ctr = get_xy_ctr_np(offsets.size(-1), score_offset, self.total_stride)
        #     fm_ctr = fm_ctr.to(offsets.device)
        # else:
        #     fm_ctr = self.fm_ctr.to(offsets.device)

        fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(fm_ctr, offsets)
        # print(bbox.shape)
        return cls_score, ctr_score, bbox

    def update_params(self):
        # super().update_params()

        q_size = self.default_hyper_params["q_size"]
        self.score_size = self.default_hyper_params["score_size"]
        self.total_stride = self.default_hyper_params["total_stride"]
        self.score_offset = (q_size - 1 - (self.score_size - 1) * self.total_stride) // 2
        self.default_hyper_params["score_offset"] = self.score_offset

        ctr = get_xy_ctr_np(self.score_size, self.score_offset, self.total_stride)
        
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False

        self._make_net()
        # self._initialize_conv()

    def _make_net(self):
        self.in_channels = self.default_hyper_params["in_channels"]
        mdim = 256

        self.cls_ctr = ConvModule(self.in_channels * 2, mdim)
        self.reg = ConvModule(self.in_channels * 2, mdim)

        # has bn, no relu
        self.cls_score = conv_bn_relu(mdim, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.ctr_score = conv_bn_relu(mdim, 1, stride=1, kszie=1, pad=0, has_relu=False)
        self.reg_offsets = conv_bn_relu(mdim, 4, stride=1, kszie=1, pad=0, has_relu=False)

    def _initialize_conv(self, ):
        # initialze head
        conv_list = []
        for m in self.cls_ctr.modules():
            if isinstance(m, nn.Conv2d):
                conv_list.append(m)
        for m in self.reg.modules():
            if isinstance(m, nn.Conv2d):
                conv_list.append(m)
        conv_list.append(self.cls_score.conv)
        conv_list.append(self.ctr_score.conv)
        conv_list.append(self.reg_offsets.conv)
        conv_classifier = [self.cls_score.conv]
        assert all(elem in conv_list for elem in conv_classifier)

        conv_weight_std = 0.0001
        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)



class STMTrack(nn.Module):
    support_phases = ["train", "memorize", "track"]

    def __init__(self, backbone_m, backbone_q, neck_m, neck_q, head, loss=None):
        super(STMTrack, self).__init__()
        self.default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)
        self.basemodel_m = backbone_m
        self.basemodel_q = backbone_q
        self.neck_m = neck_m
        self.neck_q = neck_q
        self.head = head
        self.loss = loss
        self._phase = "track"

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def memorize(self, im_crop, fg_bg_label_map):
       
        fm = self.basemodel_m(im_crop, fg_bg_label_map)
        fm = self.neck_m(fm)
        fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W
        return fm

    # def train_forward(self, training_data):
    #     memory_img = training_data["im_m"]
    #     query_img = training_data["im_q"]
    #     # backbone feature
    #     assert len(memory_img.shape) == 5
    #     B, T, C, H, W = memory_img.shape

    #     memory_img = memory_img.view(-1, C, H, W)  # no memory copy
    #     target_fg_bg_label_map = training_data["fg_bg_label_map"].view(-1, 1, H, W)

    #     fm = self.basemodel_m(memory_img, target_fg_bg_label_map)
    #     fm = self.neck_m(fm)  # B * T, C, H, W
    #     fm = fm.view(B, T, *fm.shape[-3:]).contiguous()  # B, T, C, H, W
    #     fm = fm.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W

    #     fq = self.basemodel_q(query_img)
    #     fq = self.neck_q(fq)

    #     fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = self.head(fm, fq)
    #     predict_data = dict(
    #         cls_pred=fcos_cls_score_final,
    #         ctr_pred=fcos_ctr_score_final,
    #         box_pred=fcos_bbox_final,
    #     )
    #     if self.default_hyper_params["corr_fea_output"]:
    #         predict_data["corr_fea"] = corr_fea
    #     return predict_data

    def forward(self, *args, phase=None):
        if phase is None:
            phase = self._phase
        # used during training
        # if phase == 'train':
        #     # resolve training data
        #     if self.default_hyper_params["amp"]:
        #         with torch.cuda.amp.autocast():
        #             return self.train_forward(args[0])
        #     else:
        #         return self.train_forward(args[0])

        elif phase == 'memorize':
            target_img, fg_bg_label_map = args
            fm = self.memorize(target_img, fg_bg_label_map)
            out_list = fm

        elif phase == 'track':
            assert len(args) == 2
            search_img, fm = args
            fq = self.basemodel_q(search_img)
            fq = self.neck_q(fq)  # B, C, H, W

            # print("fm size: ",fm.shape)
            # print("fq size: ",fq.shape)

            fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final = self.head(
                fm, fq, search_img.size(-1))
            
            print("fcos_cls_score_final size: ",fcos_cls_score_final.shape)
            print("fcos_ctr_score_final size: ",fcos_ctr_score_final.shape)
            print("fcos_bbox_final size: ",fcos_bbox_final.shape)
            
            # apply sigmoid
            # print("fcos_bbox_final ",fcos_bbox_final.shape)
            fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
            fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
            # apply centerness correction
            fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

            # extra = dict()
            # output
            out_list = fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final#, extra
        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    # def update_params(self):
    #     pass
    #     # self._make_convs()
    #     # self._initialize_conv()
    #     # super().update_params()

    # def _make_convs(self):
    #     head_width = self.default_hyper_params['head_width']

    #     # feature adjustment
    #     self.r_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
    #     self.c_z_k = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
    #     self.r_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)
    #     self.c_x = conv_bn_relu(head_width, head_width, 1, 3, 0, has_relu=False)

    # def _initialize_conv(self, ):
    #     conv_weight_std = self.default_hyper_params['conv_weight_std']
    #     conv_list = [
    #         self.r_z_k.conv, self.c_z_k.conv, self.r_x.conv, self.c_x.conv
    #     ]
    #     for ith in range(len(conv_list)):
    #         conv = conv_list[ith]
    #         torch.nn.init.normal_(conv.weight,
    #                               std=conv_weight_std)  # conv_weight_std=0.01

    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)


from torchvision.models.efficientnet import *



class Efficientnet_b0_M(nn.Module):
    default_hyper_params = dict(
        pretrained=True
    )

    def __init__(self, transform_input=False):
        
        super(Efficientnet_b0_M, self).__init__()
        effnet = efficientnet_b0(pretrained=True)
        self.l1 = effnet.features[0]
        self.feat_ = effnet.features[1:4]
        self.proj_fg_bg_label_map = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        torch.nn.init.normal_(self.proj_fg_bg_label_map.weight, std=0.01)

    def forward(self, x, fg_bg_label_map=None):
        bias = 255 / 2
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - bias) / bias
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - bias) / bias
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - bias) / bias
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.l1(x) + self.proj_fg_bg_label_map(fg_bg_label_map)
        x = self.feat_(x)
        return x
    
    def update_params(self):
        pass
        # super().update_params()
        # self.pretrained = self._hyper_params['pretrained']



# @VOS_BACKBONES.register
# @TRACK_BACKBONES.register
class Efficientnet_b0_Q(nn.Module):
    default_hyper_params = dict(
        pretrained=True
    )
    

    def __init__(self, transform_input=False):
        
        super(Efficientnet_b0_Q, self).__init__()
        effnet = efficientnet_b0(pretrained=True)
        self.feat_ = effnet.features[0:4]

    def forward(self, x, fg_bg_label_map=None):
        bias = 255 / 2
        x_ch0 = (torch.unsqueeze(x[:, 2], 1) - bias) / bias
        x_ch1 = (torch.unsqueeze(x[:, 1], 1) - bias) / bias
        x_ch2 = (torch.unsqueeze(x[:, 0], 1) - bias) / bias
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        x = self.feat_(x)
        return x
    
    def update_params(self):
        pass
        # super().update_params()
        # self.pretrained = self._hyper_params['pretrained']





