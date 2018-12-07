import os
import sys
import traceback
import numpy as np
import time
from imutils import get_filename_and_class, normalize_array, pil2array, imread, imshow, show_samples
import cv2 as cv2

def create_dataset(data_path, test_ratio=0.2, hot_labels=True):
    files, id2label, label2id = get_filename_and_class(data_path=data_path)
    np.random.shuffle(files)
    num_test = int(test_ratio * len(files))

    np.random.shuffle(files)
    train_files = files[num_test:]
    test_files = files[:num_test]

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    num_classes = len(id2label)

    for f in train_files:
        try:
            image = normalize_array(pil2array(image=imread(f[0])))
            image = np.reshape(image, (image.shape[0] * image.shape[1]))

            train_data.append(image)
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                train_labels.append(y)
            else:
                train_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    for f in test_files:
        try:
            image = normalize_array(pil2array(image=imread(f[0])))
            image = np.reshape(image, (image.shape[0] * image.shape[1]))
            test_data.append(image)
            if hot_labels:
                y = np.zeros(shape=num_classes, dtype=np.float32)
                y[int(f[1])] = 1.0
                test_labels.append(y)
            else:
                test_labels.append(int(f[1]))
        except Exception as _:
            traceback.print_exc(file=sys.stdout)
            continue

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    if len(train_data) == 0:
        train_data = None
        train_labels = None

    if len(test_data) == 0:
        test_data = None
        test_labels = None
    return train_data, train_labels, test_data, test_labels, id2label

if __name__ == '__main__':
    data_dir = "../caltechdataset/"
    files = get_filename_and_class(data_path=data_dir)
    show_samples(data_path=data_dir,row=10,col=20)
    print("files")
    for f in files[0]:
        print(f[0], f[1])
        img = imread(f[0])

        imshow(img)
        cv2.waitKey(100)

    # train_data, train_labels, test_data, test_labels, label_map = create_dataset(data_path=data_dir)
    # print("Classes: {}={}".format(len(label_map), label_map))
    # print("Train samples: {}, Test samples: {}".format(len(train_labels), len(test_labels)))
    # print(train_data[0])qqq