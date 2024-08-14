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

from net2 import Efficientnet_b0_M, Efficientnet_b0_Q, AdjustLayer1, STMHead1
from utils import *
from tracker import *


backbone_m = Efficientnet_b0_M()
backbone_m.update_params()
backbone_q = Efficientnet_b0_Q()
backbone_q.update_params()
neck_m = AdjustLayer1()
neck_m.update_params()
neck_q = AdjustLayer1()
neck_q.update_params()
head = STMHead1()
head.update_params()

print(neck_m)

model = STMTrack(backbone_m, backbone_q, neck_m, neck_q, head)
# model.update_params()

# Convert BatchNorm to SyncBatchNorm 
# task_model = convert_model(task_model)

# model_file = "new-epoch-19.pkl"
# model_file = "/home/meysam/test-apps/STMTrack/epoch-19_got10k.pkl"
model_file = "/home/meysam/test-apps/STMTrack/snapshots/stmtrack-effnet-got-train/epoch-0.pkl"
# model_file = "epoch-19.pkl"

extra_keys = ["r_z_k.conv.weight", "r_z_k.conv.bias", "r_z_k.bn.weight", "r_z_k.bn.bias", "r_z_k.bn.running_mean", "r_z_k.bn.running_var", "r_z_k.bn.num_batches_tracked", "c_z_k.conv.weight", "c_z_k.conv.bias", "c_z_k.bn.weight", "c_z_k.bn.bias", "c_z_k.bn.running_mean", "c_z_k.bn.running_var", "c_z_k.bn.num_batches_tracked", "r_x.conv.weight", "r_x.conv.bias", "r_x.bn.weight", "r_x.bn.bias", "r_x.bn.running_mean", "r_x.bn.running_var", "r_x.bn.num_batches_tracked", "c_x.conv.weight", "c_x.conv.bias", "c_x.bn.weight", "c_x.bn.bias", "c_x.bn.running_mean", "c_x.bn.running_var", "c_x.bn.num_batches_tracked"]

model_state_dict = torch.load(model_file,
                        map_location=torch.device("cpu"))


for e in extra_keys:
	del model_state_dict['model_state_dict'][e]

model.load_state_dict(model_state_dict['model_state_dict'])

# model.load_state_dict(model_state_dict)

pipeline_tracker = STMTrackTracker(model)
pipeline_tracker.update_params()

# dev = torch.device('cuda:0') 
dev = torch.device('cuda:0')
pipeline_tracker.set_device(dev)


g = "car1_s"
# /media/meysam/hdd/dataset/Dataset_UAV123/UAV123
path_gt = "/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/anno/UAV123/car1_s.txt" 
img_files_path = glob.glob("/media/meysam/hdd/dataset/Dataset_UAV123/UAV123/data_seq/UAV123/car1_s/*")
img_files_path.sort()

img_files = []
for i in img_files_path:
	frame = cv2.imread(i, cv2.IMREAD_COLOR)
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	img_files.append(frame)

my_file = open(path_gt)

line = my_file.readline()
line = [int(l) for l in line[:-1].split(',')]
my_file.close()


box = line
frame_num = len(img_files)
boxes = np.zeros((frame_num, 4))
boxes[0] = box
times = np.zeros(frame_num)
# my_file = open('output/'+g+'.txt','w+')
for f, img_file in enumerate(img_files):
	image = img_file
	start_time = time.time()
	if f == 0:
		pipeline_tracker.init(image, box)
	else:
		boxes[f, :] = pipeline_tracker.update(image)
		# print(pipeline_tracker._state['pscores'][-1])
		times[f] = time.time() - start_time

		# visualiation         
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		pred = boxes[f,:].astype(int)
		

		# image = cv2.resize(image,(1920,1080))
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		cv2.rectangle(image, (pred[0], pred[1]),
				(pred[0] + pred[2], pred[1] + pred[3]),
				(255,0,0), 2)
				
		line = str(pred[0])+','+str(pred[1])+','+str(pred[2])+','+str(pred[3])+'\n'
		# print(line)
		# my_file.writelines(line)
		cv2.imshow(g,image)
		print("FPS: ",1/times[f])
		# cv2.waitKey(0)        
		if cv2.waitKey(1)  == 27:
			break

# my_file.close()
cv2.destroyAllWindows()

