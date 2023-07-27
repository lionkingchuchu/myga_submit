#두 csv 파일의 결과를 앙상블 하여 최종 csv 파일을 만듭니다.

from ensemble_boxes import *
import csv
import pandas as pd
import numpy as np

root_pth = '/output/'
csv1 = root_pth+'submission1.csv'
csv2 = root_pth+'submission2.csv'

data1=pd.read_csv(csv1)
data2=pd.read_csv(csv2)
filelst=set()

for i in range(len(data1)):
    filelst.add(data1['File'][i])
filelst = list(filelst)

def box_score_label(cur):
    score = cur['Confidence'].values
    x1 = cur['X1'].values.reshape(-1,1)/1024
    y1 = cur['Y1'].values.reshape(-1,1)/1024
    x3 = cur['X3'].values.reshape(-1,1)/1024
    y3 = cur['Y3'].values.reshape(-1,1)/1024
    box = np.concatenate((x1,y1,x3,y3),axis=1)
    label = np.zeros_like(score)
    return box, score, label

def twop2fourp(two_boxes):
    four_boxes = []
    for box in two_boxes:
        x1 = box[0]
        y1 = box[1]
        x3 = box[2]
        y3 = box[3]
        x2 = x3
        y2 = y1
        x4 = x1
        y4 = y3
        four_boxes.append([float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)])
    four_boxes = np.array(four_boxes)
    four_boxes *= 1024
    return four_boxes

ensemble = []

for filename in filelst:
    cur1=data1.loc[data1['File']==filename]
    cur2=data2.loc[data2['File']==filename]

    boxes_list = []
    scores_list = []
    labels_list = []
    boxes, scores, labels = box_score_label(cur1)
    boxes_list.append(boxes)
    scores_list.append(scores)
    labels_list.append(labels)
    boxes, scores, labels = box_score_label(cur2)
    boxes_list.append(boxes)
    scores_list.append(scores)
    labels_list.append(labels)

    weights = [1,1]
    iou_thr = 0.7
    skip_box_thr = 0.15

    boxes,scores,lables = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    new_boxes = twop2fourp(boxes)
    filerow = np.array([filename for i in range(len(scores))])

    if len(scores>0):
        row = np.concatenate([filerow.reshape(-1,1),scores.reshape(-1,1),new_boxes],axis=1)
    else:
        print('no')
    for x in row:
        ensemble.append(x)

ensembledf = pd.DataFrame(ensemble, columns = ['File','Confidence','X1','Y1','X2','Y2','X3','Y3','X4','Y4'])
ensembledf.to_csv('/root/myga_submit/ensemble_result.csv',sep=',',index=False)