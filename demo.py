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

pipeline_tracker = STMTrackTracker(model)
pipeline_tracker.update_params()

# dev = torch.device('cuda:0') 
dev = torch.device('cuda:0')
pipeline_tracker.set_device(dev)


g = "car1_s"
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
                print(line)
                # my_file.writelines(line)
                # cv2.imshow(g,image)
                # # print("FPS:s ",1/times[f])
                # cv2.waitKey(0)        
                # if cv2.waitKey(1)  == 27:
                #         break

# my_file.close()
cv2.destroyAllWindows()

