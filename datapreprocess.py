import os
import json
from typing import List
import math
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

NIA_CLASSES = ['정유탱크']
CLASS_NAMES_EN = ['oil tank']

def convert_8coords_to_4coords(coords):
    x_coords = coords[0::2]
    y_coords = coords[1::2]
    
    xmin = min(x_coords)
    ymin = min(y_coords)

    xmax = max(x_coords)
    ymax = max(y_coords)

    w = xmax-xmin
    h = ymax-ymin

    return [xmin, ymin, w, h]

def convert_labels_to_objects(coords, class_ids, class_names, image_ids, difficult=0, is_clockwise=False):
    objs = list()
    inst_count = 1

    for polygons, cls_id, cls_name, img_id in tqdm(zip(coords, class_ids, class_names, image_ids), desc="converting labels to objects"):
        xmin, ymin, w, h = convert_8coords_to_4coords(polygons)
        single_obj = {}
        single_obj['difficult'] = difficult
        single_obj['area'] = w*h
        if cls_name in CLASS_NAMES_EN:
            single_obj['category_id'] = CLASS_NAMES_EN.index(cls_name)
        else:
            continue
        single_obj['segmentation'] = [[int(p) for p in polygons]]
        single_obj['iscrowd'] = 0
        single_obj['bbox'] = (xmin, ymin, w, h)
        single_obj['image_id'] = img_id
        single_obj['id'] = inst_count
        inst_count += 1
        objs.append(single_obj)

    print('objects', len(objs))
    return objs

def load_geojsons(filepath):
    """ Gets label data from a geojson label file

    :param (str) filename: file path to a geojson label file
    :return: (numpy.ndarray, numpy.ndarray ,numpy.ndarray) coords, chips, and classes corresponding to
            the coordinates, image names, and class codes for each ground truth.
    """
    jsons = glob(os.path.join(filepath, '*.json'))
    features = []
    for json_path in tqdm(jsons, desc='loading geojson files'):
        with open(json_path) as f:
            data_dict = json.load(f)
        features += data_dict['features']

    obj_coords = list()
    image_ids = list()
    class_indices = list()
    class_names = list()

    for feature in tqdm(features, desc='extracting features'):
        properties = feature['properties']
        image_ids.append(properties['image_id'].replace('PS4', 'PS3')[:-4]+'.png')
        obj_coords.append([float(num) for num in properties['object_imcoords'].split(",")])
        class_indices.append(0)
        class_names.append(properties['type_name'])

    return image_ids, obj_coords, class_indices, class_names

def geojson2coco(imageroot: str, geojsonpath: str, destfile, difficult='-1'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(CLASS_NAMES_EN):
        single_cat = {'id': idex, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 0
    with open(destfile, 'w') as f_out:
        img_files, obj_coords, cls_ids, class_names = load_geojsons(geojsonpath)
        img_id_map= {img_file:i+1 for i, img_file in enumerate(list(set(img_files)))}
        image_ids = [img_id_map[img_file] for img_file in img_files]
        objs = convert_labels_to_objects(obj_coords, cls_ids, class_names, image_ids, difficult=difficult, is_clockwise=False)
        data_dict['annotations'].extend(objs)

        for imgfile in tqdm(img_id_map, desc='saving img info'):
            imagepath = os.path.join(imageroot, imgfile)
            img_id = img_id_map[imgfile]

            img = cv2.imread(imagepath)
            try:
                height, width, c = img.shape
            except: continue
            single_image = {}
            single_image['file_name'] = imgfile
            single_image['id'] = img_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

        json.dump(data_dict, f_out)

rootfolder = './oiltank_dataset'

train_path = glob(rootfolder+'/train_images/**', recursive=True)
val_path = glob(rootfolder+'/valid_images/**', recursive=True)
exts = ('kml', 'tif')

for path in train_path:
    if any(ext in path for ext in exts):
        os.remove(path)

for path in val_path:
    if any(ext in path for ext in exts):
        os.remove(path)


geojson2coco(imageroot=os.path.join(rootfolder, 'train_images'),
                 geojsonpath=os.path.join(rootfolder, 'train_labels'),
                 destfile=os.path.join(rootfolder, 'train_labels.json'))