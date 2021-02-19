from matplotlib import pyplot as plt
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
# from keras_segmentation.models import *
from keras.models import load_model, Model

import layers
import model
# import model_deeplab3plus as DV3
# import model_FCDenseNet as FCDN
import loss

import excel
import os
import data
import numpy as np
import IBCNN
import cv2
import rasterio
import lovasz_softmax
import GAN
import postprocessing
import estimate
import classification

'''
    這個檔案可以根據我們要訓練預測的模型進行動作


    參數：
        date：存檔的檔案名稱中，日期
        training_num：訓練資料數量
        name_loss：使用的Loss名稱
        name_model：模型使用的名稱
        name：最終存的模型、預測影像、excel資料的名稱，這影響到這次程式運行會有多少個實驗要跑
        input_shape：輸入模型的資料大小
        batch：在訓練或測試的batch size
        train_flag：是否訓練，1 此階段要訓練 / 0 此階段不訓練
        test_flag：是否測試，1 此階段要測試 / 0 此階段不測試
        epochs：訓練epoch數量

    相關模型的空載入相關參數
    # model_select = model.UNet_DtoU5(block=model.RDBlocks,
        #                                 name="unet_2RD-5",
        #                                 input_size=input_shape,
        #                                 block_num=2)
        # model_select = FCDN.Tiramisu(input_shape= input_shape)
        #  model_select = hrnet.seg_hrnet(batch_size= 3, height= 256, width= 256, channel= 3, classes= 1)
    
'''


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符號
    path = path.rstrip("\\")

    # 判斷路徑是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判斷結果
    if not isExists:
        # 如果不存在則建立目錄
        print("Building the file.")
        # 建立目錄操作函式
        os.makedirs(path)
        return True
    else:
        # 如果目錄存在則不建立，並提示目錄已存在
        print("File is existing.")
        return False

if __name__ == "__main__":
    dataset = "V3.2"
    date = "20210219"
    training_num = 9338
    loss_name = "CE"
    model_name = ["UNet(2RDB8)"]
    head_name = "UNet(2RDB8)(withDataAugmentation)\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset
    # head_name = dataset + "_test\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset
    # 有改model的si程式碼，記得條回去，預測那塊
    name = [head_name + "_" + model_name[0] + "_" + loss_name[0]]

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
    # traing data
    # 要確認檔案位置
    # (train_x, train_y) = (np.load(".\\npy\\V3.2_x_1in7131.npy"),
    #                       np.load(".\\npy\\V3.2_y_1in7131.npy"))

    # (train_x_90, train_y_90) = (np.load(".\\npy\\NRR_x_1in9338.npy"),
    #                             np.load(".\\npy\\NRR_y_1in9338.npy"))
    # (train_x_180, train_y_180) = (np.load(".\\npy\\NRR_256_r180_x_1in9253.npy"),
    #                               np.load(".\\npy\\NRR_256_r180_y_1in9253.npy"))


    # testing data
    (test_x, test_y) = (np.load(".\\npy\\V3.2_x_7132in1783.npy"),
                        np.load(".\\npy\\V3.2_y_7132in1783.npy"))

    # (test_x, test_y) = (np.load(".\\npy\\NRR_x_9339in2334.npy"),
    #                    np.load(".\\npy\\NRR_y_9339in2334.npy"))
    #
    # train_x = np.concatenate([train_x_90, train_x_180], axis= 0)
    # train_y = np.concatenate([train_y_90, train_y_180], axis=0)


    #print("test x shape {}".format(test_x.shape))
    test_data_start = 1
    input_size = (256, 256, 4)
    for i in range(len(name)):
        # print("Train data shape {}\n{}".format(train_x.shape, train_y.shape))
        print("Building model.")
        model_select = load_model("E:\\allen\\data\\result\\paper_used_data\\FusionNet(withoutDataAugmentation)\\20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE.h5") # 載入要使用的模型



        epochs = 50
        batch = 3
        train_flag = 0
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        # model_select.compile(optimizer=Adam(lr=1e-4), loss='                             ', metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_size)
        # model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            model_build.train(x_train=train_x, y_train=train_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            model_build.test(x_test= test_x, y_test=test, data_start=test_data_start, batch_size=batch)