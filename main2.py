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
import cv2
import rasterio
import lovasz_softmax
import classification



if __name__ == "__main__":
    dataset = "NRR"
    date = "20201114"
    training_num = 9338
    loss_name = ["CE"]
    weights_path = [".\\result\\model_record\\NRR_test\\20201014_256(50%)_18591_NRR(degree-90&180)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR_CE.h5",
                    ".\\result\\model_record\\NRR_test\\20201113_256(50%)_18591_NRR(degree-90&180_th09_high)_testNRR_fineTuning10UNet(2RDB8-DtoU-5)-originalReduceLR_CE.h5",
                    ".\\result\\model_record\\NRR_test\\20201113_256(50%)_18591_NRR(degree-90&180_th09_low)_testNRR_fineTuning10UNet(2RDB8-DtoU-5)-originalReduceLR_CE.h5",
                    ".\\result\\model_record\\NRR_test\\20201113_256(50%)_18591_NRR(degree-90&180_th09_low)_testNRR_fineTuning10UNet(2RDB8-DtoU-5)-originalReduceLR_CE-bestweights-epoch001-loss0.10336-val_loss0.35097.h5"]

    model_name = ["FusionNetEpoch5(input3_All-HighWithNewFinetuning-LowWithNewFinetuning1113)-originalReduceLR",
                  "FusionNetEpoch5(input3_All-High-LowWithNewFinetuning1113epoch1)-originalReduceLR"]

    head_name = dataset + "_test\\" + date + "_256(50%)_" + str(18591) + "_" + dataset + "(degree-90&180_th09)_testNRR"
    name = [head_name + "_" + model_name[0] + "_" + loss_name[0]]


    # prepare data for training
    print("Load data.")
    # V3.1 Train on 6177 samples, validate on 883 samples
    # V3.2 Train on 6239 samples, validate on 892 samples
    # V3.1&3.2 Train on 12417 samples, validate on 1774 samples

    # traing data
    # traing data
    # (train_x, train_y) = (np.load(".\\npy\\V3.2_x_1in7131.npy"),
    #                       np.load(".\\npy\\V3.2_y_1in7131.npy"))
    (train_x_90, train_y_90) = (np.load(".\\npy\\NRR_x_1in9338.npy"),
                                np.load(".\\npy\\NRR_y_1in9338.npy"))
    (train_x_180, train_y_180) = (np.load(".\\npy\\NRR_256_r180_x_1in9253.npy"),
                                  np.load(".\\npy\\NRR_256_r180_y_1in9253.npy"))

    # testing data
    # (test_x, test_y) = (np.load(".\\npy\\V3.2_x_7132in1783.npy"),
    #                     np.load(".\\npy\\V3.2_y_7132in1783.npy"))
    # (test_x, test_y) = (np.load(".\\npy\\NRR_x_9339in2334.npy"),
    #                   np.load(".\\npy\\NRR_y_9339in2334.npy"))

    train_x = np.concatenate([train_x_90, train_x_180], axis=0)
    train_y = np.concatenate([train_y_90, train_y_180], axis=0)
    # excel_file = ".\\result\\data\\NRR_test\\" \
    #              "20201014_256(50%)_18591_NRR(degree-90&180)_trainDataResult_UNet(2RDB8-DtoU-5)-originalReduceLR_CE_iou.xlsx"
    # total_num = 18591
    # (train_x, train_y) = data.extract_low_result(train_x, train_y, excel_file, total_num, extract_index="iou", threshold=0.7)
    #
    # excel_file = ".\\result\\data\\NRR_test\\" \
    #              "20201014_256(50%)_18591_NRR(degree-90&180)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR_CE_iou.xlsx"
    # total_num = 2334
    # (test_x, test_y) = data.extract_low_result(test_x, test_y, excel_file, total_num, extract_index="iou",
    #                                              threshold=0.7)

    # x = np.concatenate([train_x, test_x], axis=0)
    # y = np.concatenate([train_y, test_y], axis=0)
    # (x, y) = data.remove_non_goal_img(x, y)
    # print(x.shape)
    # train_x = x[:2755]
    # train_y = y[:2755]
    # print("train x shape {}".format(train_x.shape))
    # test_x = x[2755:]
    # test_y = y[2755:]

    test_data_start = training_num + 1
    input_size = (256, 256, 4)

    # train_hsv_x = data.transHSV(train_x)
    # test_hsv_x = data.transHSV(test_x)
    # train_ycbcr_x = data.transYCbCr(train_x)
    # test_ycbcr_x = data.transYCbCr(test_x)
    # train_x = np.concatenate((train_x, data.load_data_landEdge(1, 7131)), axis=-1)
    # test_x = np.concatenate((test_x, data.load_data_landEdge(7132, 1783)), axis=-1)


    for i in range(len(name)):
        print("Building model.")
        # model_base = model.proposal_UNet_layer1(block=model.residual_block, input_size=(256, 256, 4))
        # model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))

        model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                        name="unet_2RD-5",
                                        input_size=input_size,
                                        block_num=2)
        # 1. all 2. high 3. low
        model_select.load_weights(weights_path[0])
        out1 = model_select.predict(train_x)
        model_select.load_weights(weights_path[1])
        out2 = model_select.predict(train_x)
        model_select.load_weights(weights_path[2])
        out3 = model_select.predict(train_x)
        model_select = model.Fusion_net(size=(256,256,1))
        model_select.load_weights(".\\result\\model_record\\NRR_test\\20201114_256(50%)_18591_NRR(degree-90&180_th09)_testNRR_FusionNetEpoch5(input3_All-HighWithNewFinetuning-LowWithNewFinetuning1113)-originalReduceLR_CE.h5")

        train_flag = 0
        test_flag = 1
        batch = 10
        epoch = 50

        print("compile model.")
        model_select.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['accuracy'])


        print("model building.")
        model_build = model.model(model=model_select, name=name[i], size=(256, 256, 1))

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            # training data preprocessing
            print("training processing")
            # excel_file = ".\\result\\data\\NRR_test\\" \
            #              "20201014_256(50%)_18591_NRR(degree-90&180)_trainDataResult_UNet(2RDB8-DtoU-5)-originalReduceLR_CE_iou.xlsx"
            # total_num = 18591
            # train_y = model_build.GT_data_transfer(excel_file, total_num, extract_index= "iou", threshold= 0.9)
            model_build.train(x_train=[ out1, out2, out3], y_train=train_y,
                              epochs= epoch, batch_size= batch,
                              validation_ratio= 0.125)
            # model_build.train_sampleWeights(x_train=train_x, y_train=train_y,
            #                                 batch_epoch=batch_epoch[i], epochs= epoch,
            #                                 batch_size=batch,
            #                                 threshold= th)

        if test_flag:
            print("testing processing")
            # excel_file = ".\\result\\data\\NRR_test\\" \
            #              "20201014_256(50%)_18591_NRR(degree-90&180)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR_CE_iou.xlsx"
            # total_num = 2334
            # test_y = model_build.GT_data_transfer(excel_file, total_num, extract_index="iou", threshold=0.9)
            #
            (test_x, test_y) = (np.load(".\\npy\\NRR_x_9339in2334.npy"),
                                np.load(".\\npy\\NRR_y_9339in2334.npy"))
            model_test = model.UNet_DtoU5(block=model.RDBlocks,
                                          name="unet_2RD-5",
                                          input_size=input_size,
                                          block_num=2)
            # 1. all 2. high 3. low
            model_test.load_weights(weights_path[0])
            out1 = model_test.predict(test_x)
            model_test.load_weights(weights_path[1])
            out2 = model_test.predict(test_x)
            model_test.load_weights(weights_path[2])
            out3 = model_test.predict(test_x)

            model_build.test(x_test=[out1, out2, out3], y_test=test_y, data_start=test_data_start, batch_size=batch)