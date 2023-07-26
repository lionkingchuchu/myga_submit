import pickle
import pandas as pd
import csv

path='/output/results1.pkl'
outdir='/output/'
data = pd.read_pickle(path)
csvfilename = 'submission1'

def buildrow(objname, bbox, score):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[1]
    x3 = bbox[2]
    y3 = bbox[3]
    x4 = bbox[0]
    y4 = bbox[3]
    return [objname, float(score), float(x1), float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)]

writelst = [['File','Confidence','X1','Y1','X2','Y2','X3','Y3','X4','Y4']]
for i in range(len(data)):
    objname = list(data[i]['img_path'].split('/'))[6][:8]
    num_bbox = len(data[i]['pred_instances']['bboxes'])

    for j in range(num_bbox):
        row = buildrow(objname, data[i]['pred_instances']['bboxes'][j], data[i]['pred_instances']['scores'][j])
        writelst.append(row)

f=open(f"{outdir}{csvfilename}.csv", 'w', newline='')
for row in writelst:
    writer = csv.writer(f)
    writer.writerows([row])
f.close()


path='/output/results2.pkl'
outdir='/output/'
data = pd.read_pickle(path)
csvfilename = 'submission2'

def buildrow(objname, bbox, score):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[1]
    x3 = bbox[2]
    y3 = bbox[3]
    x4 = bbox[0]
    y4 = bbox[3]
    return [objname, float(score), float(x1), float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)]

writelst = [['File','Confidence','X1','Y1','X2','Y2','X3','Y3','X4','Y4']]
for i in range(len(data)):
    objname = list(data[i]['img_path'].split('/'))[6][:8]
    num_bbox = len(data[i]['pred_instances']['bboxes'])

    for j in range(num_bbox):
        row = buildrow(objname, data[i]['pred_instances']['bboxes'][j], data[i]['pred_instances']['scores'][j])
        writelst.append(row)

f=open(f"{outdir}{csvfilename}.csv", 'w', newline='')
for row in writelst:
    writer = csv.writer(f)
    writer.writerows([row])
f.close()