import numpy as np
import os
import cv2
import model
from keras.models import load_model

from keras.optimizers import *
# import model_deeplab3plus as DV3
# import model_FCDenseNet as FCDN
import classifiation
import postprocessing
import excel
import estimate
import data

'''
    這個檔案訓練跟測試 JudgeNet

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
        threshold：將資料分成兩類的 IoU門檻值

'''

if __name__ == "__main__":

    dataset = "V3.2"
    date = "20201216"
    training_num = 9338
    loss_name = "CE"
    name_model = ["JudgeAlexNet(withoutDataAugmentation)"]

    threshold = np.array([0.9])
    head_name = dataset + "_test\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset
    # 有改model的si程式碼，記得條回去，預測那塊
    name = [head_name + "_" + model_name[0] + "_" + loss_name[0]]

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
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
        input_shape = (256, 256, 3)
        model_select = classifiation.Alexnet(input_shape, output_class= 2)
        model_select.load_weights(".\\result\\model_record\\20201216_256_51984(th09)_JudgeAlexNet_CE.h5")

        epochs = 30
        batch = 10
        train_flag = 0
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="categorical_crossentropy", metrics=['accuracy'])
        # model_select.compile(optimizer=Adam(lr=1e-4), loss='                             ', metrics=['accuracy'])

        print("Model building.")
        model_build = model.Judgement_model(model_select, name[i], input_shape= input_shape, classes = 2)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Start training.")
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CETrainData_iou.xlsx"
            total_num = 51984
            train_y = model_build.GT_data_transfer(excel_file, total_num, extract_index= "iou", threshold= 0.9)
            model_build.train(x=train_x, y=train_y, batch_size=batch, epochs= epochs)

        if test_flag:
            print("Start testing.")
            excel_file = ".\\result\\data\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE_iou.xlsx"
            total_num = 12996
            test_y = model_build.GT_data_transfer(excel_file, total_num, extract_index="iou", threshold=0.9)
            model_build.test(x= test_x, y=test_y, data_start=test_data_start, batch_size=batch)