import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.image as mpimg

from DNN import *


def load_datad_set_from_folder(folder):
    filesList = os.listdir(folder)
    df = pd.DataFrame()
    for file in filesList:
        im = mpimg.imread(folder+ file)
        im2 = np.ravel(im).reshape((1,1024))
        word_lable = file.split("_")[0]
        if word_lable == 'neg':
            lable = 0
        else:
            lable = 1
        dfRow  = pd.DataFrame({'data': [im2] , 'lable' : [lable]})
        df  = df.append(dfRow)
    return df
#
# if __name__ == '__main__':
#     # load_datad_set_from_folder("C:\\Users\\Yotam\\Desktop\\MS_Dataset_2019\\training\\")
#     dnn = DNN(512)
#     dnn.train('C:\\Users\\Yotam\\Desktop\\MS_Dataset_2019\\training\\',300)