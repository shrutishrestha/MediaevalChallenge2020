# -*- coding: utf-8 -*-
import monai
import os
import cv2
from glob import glob
import torch
import csv
from itertools import zip_longest
import numpy as np


def create_dir(folderpath):
    try:
        if not os.path.exists(folderpath):
            os.makedirs(folderpath)
    except OSError:
        print(f"Error: creating directory with name {folderpath}")


def create_file(filepath):
    try:
        if not os.path.exists(filepath):
            print("creating file", filepath)
            open(filepath, 'w').close()
    except OSError:
        print(f"Error: creating file with name {filepath}")


def read_data(x, y):
    "read image and mask from the given path"
    image = cv2.imread(x, cv2.IMREAD_COLOR)
    mask = cv2.imread(y, cv2.IMREAD_COLOR)
    return image, mask


def load_data(path):
    images_path = os.path.join(path, "images/")
    masks_path = os.path.join(path, "masks/")
    images = glob(images_path)
    masks = glob(masks_path)
    return images, masks


def load_model_weight(path, model):
    model.load_state_dict(torch.load(path)['state_dict'])
    model = model.eval()
    return model


def write_to_csv(path, parameter_dict, mode="w"):
    parameter_names = parameter_dict.keys()
    parameter_value_list = list()

    for parameter in parameter_names:
        parameter_value_list.append(parameter_dict[parameter])
    parameter_lists = zip_longest(*parameter_value_list, fillvalue='')

    with open(path, mode) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(parameter_names)
        csv_writer.writerows(parameter_lists)
    f.close()


def write_heading_in_csv(path, head_values, mode="w"):
    with open(path, mode) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(head_values)
    f.close()


def append_data_csv(path, value_list, mode="a"):
    with open(path, mode) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(value_list)
    f.close()






