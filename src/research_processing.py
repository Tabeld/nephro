import json
import os
import pydicom

import numpy as np
from pydicom.pixel_data_handlers import apply_voi_lut, apply_modality_lut

from Area import Area
from Vector import Vector

def get_research_information(path):
    files = os.listdir(path)
    segmentation_exist = True
    if not os.path.exists(path + "/data/data.json"):
        segmentation_exist = False
        data = None
    else:
        with open(path + "/data/data.json", 'r') as json_file:
            data = json.load(json_file)
    segmentation_list = list()
    data_list = list()
    for file in files:
        file_path = os.path.join(path, file)
        if file_path.endswith(".dcm"):
            dicom_data = pydicom.dcmread(file_path)
            if dicom_data.SeriesDescription.upper() not in ['VEN', 'VENOUS', 'PORTAL']:
                continue
            image = apply_voi_lut(apply_modality_lut(dicom_data.pixel_array, dicom_data), dicom_data)
            if segmentation_exist:
                marking = get_marking(data, file)
            else:
                marking = None
            if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                image = np.amax(image) - image

            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            if image.shape[0] == image.shape[1]:
                data_list.append(image)
                segmentation_list.append(process_file(image, marking))
            print(path + "/" + file), "обработан\n"
    data_list = np.array(data_list)
    segmentation_list = np.array(segmentation_list)
    return data_list, segmentation_list

def process_file(image, marking):
    h = len(image)
    w = len(image[0])
    res = np.zeros((h, w))
    if marking is None:
        return res
    for i in range(h):
        for j in range(w):
            point = Vector(j, i)
            for area in marking:
                if area.pointInsideShape(point):
                    if i == 0 and j == 0:
                        pass
                    res[i, j] = 1
                    continue
    return res

def get_marking(data, file):

    marking = data['files'][file]['marking']
    if len(marking) != 0:
        load_areas = list()
        for i in range(len(marking)):
            areas_from_file = marking[i]
            new_area = Area()
            new_area.set_mark(areas_from_file['mark'])
            new_area.add_points(areas_from_file['points'])
            load_areas.append(new_area)
        return load_areas
    return []
