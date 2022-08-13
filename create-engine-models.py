#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import glob
import torch
# import torch.nn as nn 
import time
import math
# import torch.nn.functional as F
from net import *
from utils import *
from tracker import *
import onnx
import tensorrt as trt
import torch
from torch2trt import torch2trt
import os


# In[11]:


backbone_m = Inception3_M()
backbone_m.update_params()
backbone_q = Inception3_Q()
backbone_q.update_params()
neck_m = AdjustLayer()
neck_m.update_params()
neck_q = AdjustLayer()
neck_q.update_params()
head = STMHead()
head.update_params()

model = STMTrack(backbone_m, backbone_q, neck_m, neck_q, head)
# model.update_params()

# Convert BatchNorm to SyncBatchNorm 
# task_model = convert_model(task_model)
model_file = "new-epoch-19.pkl"

# model_file = "epoch-19.pkl"
model_state_dict = torch.load(model_file,
                        map_location=torch.device("cpu"))

# model.load_state_dict(model_state_dict['model_state_dict'])
model.load_state_dict(model_state_dict)


# In[12]:


class Basemodel_Q(nn.Module):
    def __init__(self):
        super(Basemodel_Q, self).__init__()
        self.backbone_q = model.basemodel_q
        self.neck_q = model.neck_q
    def forward(self, search_img):
        fq = self.backbone_q(search_img)
        fq = self.neck_q(fq)
        return fq


# In[13]:


net = Basemodel_Q()


# In[14]:


ONNX_FILE_PATH = "backbone_q.onnx"


# In[15]:


input = torch.randn(1,3,289,289).cuda()
net.eval()
net.cuda()
# torch.onnx.export(net, input, ONNX_FILE_PATH, input_names=["search_img"], output_names=["fq"], export_params=True)

MODEL_NAME = "backbone_q"

input = torch.ones(1,3,289,289).cuda()

model_trt = torch2trt(
    net,
    [input],
    input_names=["search_img"],
    output_names=["fq"],
    fp16_mode=True,
    log_level=trt.Logger.INFO,
    max_workspace_size=(1 << 32),
    max_batch_size=1,
    use_onnx = False,
)

torch.save(model_trt.state_dict(),  MODEL_NAME + ".pth")
engine_file = MODEL_NAME+".engine"
with open(engine_file, "wb") as f:
    f.write(model_trt.engine.serialize())


# In[16]:





# In[17]:





# In[18]:


class Memorize(nn.Module):
    def __init__(self):
        super(Memorize, self).__init__()
        self.backbone_m = model.basemodel_m
        self.neck_m = model.neck_m
    def forward(self, im_crop, fg_bg_label_map):
        fm = self.backbone_m(im_crop, fg_bg_label_map)
        fm = self.neck_m(fm)
        fm = fm.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # B, C, T, H, W
        return fm


# In[19]:


net = Memorize()#(model.basemodel_m,model.neck_m)


# In[20]:


input = torch.ones(1,3,289,289).cuda()
fg_bg = torch.ones(1,1,289,289).cuda()
net.eval()
net.cuda()
fm = net(input,fg_bg)


# In[25]:


ONNX_FILE_PATH = "memorize.onnx"
input = torch.randn(1,3,289,289).cuda()
fg_bg = torch.randn(1,1,289,289).cuda()
net.eval()
net.cuda()
# torch.onnx.export(net, (input,fg_bg), ONNX_FILE_PATH, input_names=["img","fg_bg_label_map"], output_names=["fm"], export_params=True)
MODEL_NAME = "memorize"

input = torch.ones(1,3,289,289).cuda()
fg_bg = torch.ones(1,1,289,289).cuda()

model_trt = torch2trt(
    net,
    [input,fg_bg],
    input_names=["img","fg_bg_label_map"],
    output_names=["fm"],
    fp16_mode=True,
    log_level=trt.Logger.INFO,
    max_workspace_size=(1 << 32),
    max_batch_size=1,
    use_onnx = False,
)

torch.save(model_trt.state_dict(),  MODEL_NAME + ".pth")
engine_file = MODEL_NAME+".engine"
with open(engine_file, "wb") as f:
    f.write(model_trt.engine.serialize())


# In[26]:


fm = net(input,fg_bg)


# In[28]:


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


# In[29]:


hps = dict(
        total_stride=8,
        score_size=25,
        score_offset=-1,
        test_lr=0.95,
        penalty_k=0.04,
        window_influence=0.21,
        windowing="cosine",
        m_size=289,
        q_size=289,
        min_w=10,
        min_h=10,
        phase_memorize="memorize",
        phase_track="track",
        corr_fea_output=False,
        num_segments=4,
        confidence_threshold=0.6,
        gpu_memory_threshold=-1,
        search_area_factor=4.0,
        visualization=False )


# In[138]:


class Head(nn.Module):
    def __init__(self,head):
        super(Head, self).__init__()
        self.head = head
        self.total_stride = 8
        score_offset = (hps["q_size"] - 1.0 - (hps["score_size"] - 1) * hps["total_stride"]) // 2.0
        self.fm_ctr = get_xy_ctr_np(hps["score_size"], score_offset, hps["total_stride"])
    def forward(self,fm,fq):
#         q_size = 299
#         fm = fm.permute(0, 2,1, 3, 4)
        y = self.head.memory_read(fm, fq)
        cls_score, ctr_score, offsets = self.head.solve(y)
        
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)

        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)

        offsets = torch.exp(self.head.si * offsets + self.head.bi) * self.total_stride

        
        fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(fm_ctr, offsets)
        fcos_cls_score_final = cls_score
        fcos_ctr_score_final = ctr_score
        fcos_bbox_final = bbox

        
        fcos_cls_prob_final = torch.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = torch.sigmoid(fcos_ctr_score_final)
        # apply centerness correction
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final
        
        return fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final


# In[139]:


net = Head(model.head)
net.eval()
net.cuda()
ONNX_FILE_PATH = "head.onnx"
fm = torch.randn(1, 512, 6, 25, 25).cuda()
fq = torch.randn(1, 512, 25, 25).cuda()


# In[140]:


# torch.onnx.export(net, (fm,fq), 
#                   ONNX_FILE_PATH, input_names=["fm","fq"], 
#                   output_names=["fcos_score_final", "fcos_bbox_final",
#                                 "fcos_cls_prob_final","fcos_ctr_prob_final"]
#                   , export_params=True)

MODEL_NAME = "head"

fm = torch.ones(1, 512, 6, 25, 25).cuda()
fq = torch.ones(1, 512, 25, 25).cuda()

model_trt = torch2trt(
    net,
    [fm,fq],
    input_names=["fm","fq"],
    output_names=["fcos_score_final", "fcos_bbox_final",
                                "fcos_cls_prob_final","fcos_ctr_prob_final"],
    fp16_mode=True,
    log_level=trt.Logger.INFO,
    max_workspace_size=(1 << 32),
    max_batch_size=1,
    use_onnx = False,
)

torch.save(model_trt.state_dict(),  MODEL_NAME + ".pth")
engine_file = MODEL_NAME+".engine"
with open(engine_file, "wb") as f:
    f.write(model_trt.engine.serialize())

