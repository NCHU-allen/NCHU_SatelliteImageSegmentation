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
import gc
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
    dataset = "NR"
    date = "2021012"
    loss_name = ["CE"]

    model_name = ["UNet(2RDB8-DtoU-5)"]

    train_num = [9597,
                 9545,
                 9583,
                 9610,
                 9648,
                 9614]
    valid_num = [1369,
                 1364,
                 1369,
                 1373,
                 1378,
                 1373]
    dataset_total_num = [13684,
                         13636,
                         13690,
                         13728,
                         13782,
                         13734]
    dataset_root_path = "E:\\allen\\data\\dataset"

    dataset_file = ["dataset_NR_256_str12",
                    "dataset_NR_256_str13",
                    "dataset_NR_256_str14",
                    "dataset_NR_256_str23",
                    "dataset_NR_256_str24",
                    "dataset_NR_256_str34"]
    dataset_base_file = ["dataset_NR_256_dir1_r270",
                         "dataset_NR_256_dir2_r90",
                         "dataset_NR_256_dir3_r90",
                         "dataset_NR_256_dir4_r180"]

    head_name = [dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str12)",
                 dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str13)",
                 dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str14)",
                 dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str23)",
                 dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str24)",
                 dataset + "_test\\" + date + "_256(50%)_" + str(train_num[0]) + "_" + dataset + "(str34)"]
    # 有改model的si程式碼，記得條回去，預測那塊
    name = [head_name[0] + "_" + model_name[0] + "_" + loss_name[0],
            head_name[1] + "_" + model_name[0] + "_" + loss_name[0],
            head_name[2] + "_" + model_name[0] + "_" + loss_name[0],
            head_name[3] + "_" + model_name[0] + "_" + loss_name[0],
            head_name[4] + "_" + model_name[0] + "_" + loss_name[0],
            head_name[5] + "_" + model_name[0] + "_" + loss_name[0]]
    input_size = (256, 256, 4)

    # 先進行資料庫生成
    # 12
    # data.dataset_resave(dataset_path_list= [dataset_root_path + "\\" + dataset_base_file[0],
    #                                         dataset_root_path + "\\" + dataset_base_file[1]],
    #                     dataset_name_list=[dataset_base_file[0],
    #                                        dataset_base_file[1]],
    #                     dataset_totalNum_list=[6796, 6888],
    #                     concate_dataset_path= dataset_root_path + "\\" + dataset_file[0])
    # # 13
    # data.dataset_resave(dataset_path_list=[dataset_root_path + "\\" + dataset_base_file[0],
    #                                        dataset_root_path + "\\" + dataset_base_file[2]],
    #                     dataset_name_list=[dataset_base_file[0],
    #                                        dataset_base_file[2]],
    #                     dataset_totalNum_list=[6796, 6840],
    #                     concate_dataset_path=dataset_root_path + "\\" + dataset_file[1])
    # 14
    # data.dataset_resave(dataset_path_list=[dataset_root_path + "\\" + dataset_base_file[0],
    #                                        dataset_root_path + "\\" + dataset_base_file[3]],
    #                     dataset_name_list=[dataset_base_file[0],
    #                                        dataset_base_file[3]],
    #                     dataset_totalNum_list=[6796, 6894],
    #                     concate_dataset_path=dataset_root_path + "\\" + dataset_file[2])
    # # 23
    # data.dataset_resave(dataset_path_list=[dataset_root_path + "\\" + dataset_base_file[1],
    #                                        dataset_root_path + "\\" + dataset_base_file[2]],
    #                     dataset_name_list=[dataset_base_file[1],
    #                                        dataset_base_file[2]],
    #                     dataset_totalNum_list=[6888, 6840],
    #                     concate_dataset_path=dataset_root_path + "\\" + dataset_file[3])
    # # 24
    # data.dataset_resave(dataset_path_list=[dataset_root_path + "\\" + dataset_base_file[1],
    #                                        dataset_root_path + "\\" + dataset_base_file[3]],
    #                     dataset_name_list=[dataset_base_file[1],
    #                                        dataset_base_file[3]],
    #                     dataset_totalNum_list=[6888, 6894],
    #                     concate_dataset_path=dataset_root_path + "\\" + dataset_file[4])
    # # 34
    # data.dataset_resave(dataset_path_list=[dataset_root_path + "\\" + dataset_base_file[2],
    #                                        dataset_root_path + "\\" + dataset_base_file[3]],
    #                     dataset_name_list=[dataset_base_file[2],
    #                                        dataset_base_file[3]],
    #                     dataset_totalNum_list=[6840, 6894],
    #                     concate_dataset_path=dataset_root_path + "\\" + dataset_file[5])

    for i in range(len(name)):
        if i != 4:
            continue
        print("Building model.")
        model_select = model.UNet_DtoU5(block= model.RDBlocks,
                                        name= "unet_2RD-5",
                                        input_size= input_size,
                                        block_num= 2)
        model_select.load_weights(".\\result\\model_record\\NR_test\\2021012_256(50%)_9597_NR(str34)_UNet(2RDB8-DtoU-5)_CE.h5")
        print("compile model.")
        model_select.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("model building.")
        model_build = model.model(model=model_select, name=name[i], size=input_size)

        print("Select train model：{}".format(model_build.name))

        train_flag = 0
        test_flag = 1
        batch = 3
        epoch = 50

        print("start train.")
        if train_flag:
            (train_x, train_y) = data.load_data(dataset=dataset_file[i], start_num=1, total_num=train_num[i],
                                                size=input_size)
            (valid_x, valid_y) = data.load_data(dataset=dataset_file[i], start_num=train_num[i] + 1,
                                                total_num=valid_num[i],
                                                size=input_size)
            model_build.train(x_train=train_x, y_train=train_y, valid_x=valid_x, valid_y=valid_y, epochs=epoch,
                              batch_size=batch)
            # model_build.train_generator(dataset= dataset_file[i], train_num= train_num[i], valid_num= valid_num[i], epochs= epoch, batch_size= batch)

        print("start test.")
        if test_flag:
            start_num = train_num[i] + valid_num[i] + 1
            total_num = dataset_total_num[i] - train_num[i] - valid_num[i]
            (x, y) = data.load_data(dataset=dataset_file[i], start_num=start_num, total_num=total_num, size=input_size)
            test_data_start = start_num
            model_build.test(x_test=x, y_test=y, data_start=test_data_start, batch_size=batch)