import glob
import numpy as np
folders = glob.glob('train/*')
from sklearn.model_selection import train_test_split
records = []
for folder in folders:
    files = glob.glob(folder+"/*.jpg")
    labels = [e.split('/')[1] for e in files]
    train, valid = train_test_split(range(len(files)), test_size=0.2, random_state=1983)
    mask = np.zeros(len(files))
    for j in train:
        mask[j] = 1
    for i, label in enumerate(labels):
        d = dict(label=label, file=files[i], train=mask[i])
        records.append(d)
import pandas as pd

from keras_preprocessing.image import ImageDataGenerator


import json
bb_json = {}
anno_classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NOF']
for c in anno_classes:
    j = json.load(open('bbox/{}_labels.json'.format(c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]
#print(bb_json.keys())
count=0
tot=0
keys = bb_json.keys()
records2 = []
for r in records:
    tot +=1
    name = r['file'].split('/')[-1]
    if not name in keys:
        count += 1
        #print(r['file'])
        r['bbox'] = None
    else:
        bbox = bb_json[name]
        r['x'] = bbox['x']
        r['y'] = bbox['y']
        r['width'] = bbox['width']
        r['height'] = bbox['height']
        records2.append(r)
print("nobbox", count, tot)
print("rec5", records2[:5])
df = pd.DataFrame.from_records(records2)
dftrain = df[df.train==1][['file', 'label', 'x', 'y', 'width', 'height']]
dfvalid = df[df.train==0][['file', 'label', 'x', 'y', 'width', 'height']]

dftrain.to_csv("tv_train.csv", index=False, header=True)
dfvalid.to_csv("tv_valid.csv", index