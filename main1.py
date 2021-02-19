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
    這個檔案包含了個別模型的訓練跟測試，可以先選擇要使用的模型，包含了權重位置，進行測試

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
    dataset = "NRR" # 所使用的資料庫為哪個，這影響到後面存檔位置
    date = "20201226"   # 存檔
    training_num = 9338
    loss_name = ["CE"]
    model_name = ["IBCNN-UNet(2RDB8-DtoU-5)"]
    head_name = dataset + "_test\\" + date + "_256(50%)_" + str(18591) + "_" + dataset+ "(degree-90&180)_testNRR"
    # 有改model的si程式碼，記得條回去，預測那塊
    name = [head_name + "_" + model_name[0] + "_" + loss_name[0]]


    # prepare data for training
    print("Load data.")
    # V3.1 Train on 6177 samples, validate on 883 samples
    # V3.2 Train on 6239 samples, validate on 892 samples
    # V3.1&3.2 Train on 12417 samples, validate on 1774 samples

    # traing data
    # 要確認檔案位置
    # (train_x, train_y) = (np.load(".\\npy\\V3.2_x_1in7131.npy"),
    #                       np.load(".\\npy\\V3.2_y_1in7131.npy"))

    (train_x_90, train_y_90) = (np.load(".\\npy\\NRR_x_1in9338.npy"),
                                np.load(".\\npy\\NRR_y_1in9338.npy"))
    (train_x_180, train_y_180) = (np.load(".\\npy\\NRR_256_r180_x_1in9253.npy"),
                                  np.load(".\\npy\\NRR_256_r180_y_1in9253.npy"))


    # testing data
    # (test_x, test_y) = (np.load(".\\npy\\V3.2_x_7132in1783.npy"),
    #                     np.load(".\\npy\\V3.2_y_7132in1783.npy"))

    (test_x, test_y) = (np.load(".\\npy\\NRR_x_9339in2334.npy"),
                       np.load(".\\npy\\NRR_y_9339in2334.npy"))

    train_x = np.concatenate([train_x_90, train_x_180], axis= 0)
    train_y = np.concatenate([train_y_90, train_y_180], axis=0)


    #print("test x shape {}".format(test_x.shape))
    test_data_start = training_num + 1
    input_size = (256, 256, 4)

    for i in range(len(name)):
        print("Building model.")

        # 選擇要使用的模型 / 載入模型，可以選擇使用載入模型或載入權重
        model_select = model.UNet_DtoU5(block= model.RDBlocks,
                                        name= "unet_2RD-5",
                                        input_size= input_size,
                                        block_num= 2)
        # model_select = load_model("已訓練完成之模型位置")
        # model_select.load_weights("已訓練完成之模型權重位置")

        print("compile model.")
        model_select.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        # model_judge.compile(optimizer=Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
        # model_select.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

        print("model building.")
        model_build = model.model(model=model_select, name=name[i], size=input_size)

        print("Select train model：{}".format(model_build.name))

        train_flag = 0
        test_flag = 1
        batch = 10
        epoch = 50

        print("start train.")
        if train_flag:
            model_build.train(x_train= train_x, y_train= train_y, epochs= epoch, batch_size= batch)


        print("start test.")
        if test_flag:
            model_build.test(x_test=test_y, y_test=test_y, data_start=test_data_start, batch_size=batch)
