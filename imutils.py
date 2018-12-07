import os
import traceback
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sys
import cv2

def pil2array(image):
    return np.asarray(image)

def array2pil(image):
    return Image.fromarray(np.uint8(image)).convert('RGB')

def imread(filename, cv=True):
    if cv:
        image = cv2.imread(filename)
    else:
        image = Image.open(filename)
    return image

def im2bw(image):
    return image.convert('1')

def rgb2gray(image):
    return image.convert('L')

def imresize(image, size):
    return image.resize(size, Image.ANTIALIAS)


def normalize_array(image):
    return image / 255.0

def imshow_array(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def imshow(image, winname="Image", cv=True):
    if cv:
        cv2.imshow(winname, image)
    else:
        plt.imshow(np.asarray(image), cmap='gray')
        plt.show()


def show_samples(data_path, row=3, col=10):
    files, id2label, label2id = get_filename_and_class(data_path=data_path)
    np.random.shuffle(files)

    v = None
    i = 0
    for r in range(row):
        h = None
        for c in range(col):
            image = rgb2gray(imresize(image=imread(files[i][0], cv=False), size=(128, 128)))
            image = normalize_array(pil2array(image))
            i += 1
            if h is None:
                h = image.copy()
                h = np.hstack((h, np.zeros((128, 1))))
            else:
                h = np.hstack((h, image, np.zeros((128, 1))))

        if v is None:
            v = h.copy()
            v = np.vstack((v, np.zeros((1, h.shape[1]))))
        else:
            v = np.vstack((v, h, np.zeros((1, h.shape[1]))))
    imshow_array(v)
    return v


def get_filename_and_class(data_path, max_classes=0, min_samples_per_class=0):
    """Returns a list of filename and inferred class names.
  Args:
      :param data_path: A directory containing a set of subdirectories representing class names. Each subdirectory should contain PNG or JPG encoded images.
      :param min_samples_per_class:
    :param max_classes:
    data_path:
  Returns:
    A list of image file paths, relative to `data_path` and the list of
    subdirectories, representing class names.

  """
    folders = [name for name in os.listdir(data_path) if
               os.path.isdir(os.path.join(data_path, name))]

    if len(folders) == 0:
        raise ValueError(data_path + " does not contain valid sub directories.")
    directories = []
    for folder in folders:
        directories.append(os.path.join(data_path, folder))

    folders = sorted(folders)
    label2id = {}

    i = 0
    c = 0
    total_files = []
    for folder in folders:
        dir = os.path.join(data_path, folder)
        files = os.listdir(dir)
        if min_samples_per_class > 0 and len(files) < min_samples_per_class:
            continue

        for file in files:
            path = os.path.join(dir, file)
            total_files.append([path, i])
        label2id[folder] = i
        i += 1

        if 0 < max_classes <= c:
            break
        c += 1

    id2label = {v: k for k, v in label2id.items()}
    return np.array(total_files), id2label, label2id
