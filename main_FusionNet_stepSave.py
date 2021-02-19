import os
import cv2
import model
from keras.models import load_model
import numpy as np

from keras.optimizers import *
# import model_deeplab3plus as DV3
# import model_FCDenseNet as FCDN
import classification
import postprocessing
import excel
import estimate
import data
import gc

'''
    這個檔案訓練跟測試 Fused RDB U-Net

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

if __name__ == "__main__":
    dataset = "V3.2"
    date = "20210219"
    training_num = 9338
    loss_name = "CE"
    model_name = ["FusionNet(sigmoid)",
                  "FusionNet(relu)"]
    head_name = "FusionNet(withoutDataAugmentation)\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset
    # head_name = dataset + "_test\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset
    # 有改model的si程式碼，記得條回去，預測那塊
    name = [head_name + "_" + model_name[0] + "_" + loss_name[0],
            head_name + "_" + model_name[1] + "_" + loss_name[0]]

    activation = ["sigmoid",
                  "relu"]
    FusionNet_model_path = ["E:\\allen\\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201124_256(50%)_18591_NRR(degree-90&180)_testNRR_MultiParallel-JudgeAlexnet(HighLow)-All-originalReduceLR_CE(sigmoid).h5",
                            "E:\\allen\\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201117_256(50%)_18591_NRR(degree-90&180)_testNRR_MultiParallel-JudgeAlexnet(HighLow)-All-originalReduceLR_CE(relu).h5"]


    test_data_start = training_num + 1
    input_shape = (256, 256, 4)
    # processing
    processing_index = ["train",
                        "valid",
                        "test"]

    # model_weights
    # 設定第一分支模型的權重檔案位置
    main_seg_weights = "E:\\allen\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201014_256(50%)_18591_NRR(degree-90&180)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR_CE(FirstBranch).h5"
    # 設定JudgeNet的權重檔案位置
    sub_judge_weights = "E:\\allen\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201106_256(50%)_18591_NRR(degree-90&180_th09)_testNRR_Judgement-Alexnet(classification)-originalReduceLR_CE(JudgeNet).h5"
    # 設定第二分支針對大於IOU門檻訓練得到的模型權重檔案位置
    sub_segHigh_weights = "E:\\allen\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201109_256(50%)_18591_NRR(degree-90&180_th09_high)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR_CE.h5"
    # 設定第二分支針對小於IOU門檻訓練得到的模型權重檔案位置
    sub_segLow_weights = "E:\\allen\data\\result\\paper_used_data\\FusionNet(withDataAugmentation)\\20201111_256(50%)_18591_NRR(degree-90&180_th09_low)_testNRR_fineTuning10UNet(2RDB8-DtoU-5)-originalReduceLR_CE.h5"

    # npy 檔案名稱，相關資料的位置
    npy_train_x = ".\\npy\\total-70658_1-1_train_x_49461.npy"
    npy_train_y = ".\\npy\\total-70658_1-1_train_y_49461.npy"
    npy_valid_x = ".\\npy\\total-70658_1-1_valid_x_7065.npy"
    npy_valid_y = ".\\npy\\total-70658_1-1_valid_y_7065.npy"
    npy_test_x = "E:\\allen\\data\\npy\\V3.2_x_7132in1783.npy"
    npy_test_y = "E:\\allen\\data\\npy\\V3.2_y_7132in1783.npy"

    # 第一分支的預測結果檔案名稱
    npy_save_main_train_predict_y = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"-mainpipe_train_predict_y.npy"
    npy_save_main_valid_predict_y = ".\\npy\\20210118_256_49461_UNet(2-RDB8)_CE-mainpipe_valid_predict_y.npy"
    npy_save_main_test_predict_y = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"-mainpipe_test_predict_y.npy"

    # 第二分支JudgeNet針對資料的預測結果檔案名稱
    npy_save_sub_train_dist = ".\\npy\\20201216_256_49461(th05)_JudgeAlexNet_CE-Judge-train-distData.npy"
    npy_save_sub_valid_dist = ".\\npy\\20201216_256_49461(th05)_JudgeAlexNet_CE-Judge-valid-distData.npy"
    npy_save_sub_test_dist = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"-Judge-test-distData.npy"

    # 第二分支分割模型的預測結果檔案名稱
    npy_save_sub_train_high_predict_y = ".\\npy\\20201128_256_51984(05high)_UNet(2RDB8-DtoU-5)_CE_train-HighY.npy"
    npy_save_sub_train_low_predict_y = ".\\npy\\220201128_256_51984(05low)_UNet(2RDB8-DtoU-5)_CE_train-LowY.npy"
    npy_save_sub_valid_high_predict_y = ".\\npy\\20201128_256_51984(05high)_UNet(2RDB8-DtoU-5)_CE_valid-HighY.npy"
    npy_save_sub_valid_low_predict_y = ".\\npy\\220201128_256_51984(05low)_UNet(2RDB8-DtoU-5)_CE_valid-LowY.npy"
    npy_save_sub_test_high_predict_y = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"_test-HighY.npy"
    npy_save_sub_test_low_predict_y = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"_test-LowY.npy"

    # 第二分支分割模型的預測結果合併後的檔案名稱
    npy_save_sub_train_predict_y = ".\\npy\\220201128_256_51984(05)_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_predictY.npy"
    npy_save_sub_valid_predict_y = ".\\npy\\220201128_256_51984(05)_UNet(2RDB8-DtoU-5)_CE-predictValidData(V1)-subpipe_predictY.npy"
    npy_save_sub_test_predict_y = ".\\npy\\npy_FusionNet(withDataAugmentation)-V3.2\\" + date + "_256(50%)_" + str(training_num) + "_" + dataset +"-subpipe_predictY.npy"

    print("Loading data.")
    # Train on 25145 samples, validate on 3593 samples
    model_segmentation = model.UNet_DtoU5(block=model.RDBlocks,
                                          name="unet_2RD-5",
                                          input_size=input_shape,
                                          block_num=2)
    model_segmentation.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

    # # 主支線生成訓練測試x資料
    # model_segmentation.load_weights(main_seg_weights)
    # mainpipe_train_x = model_segmentation.predict(np.load(npy_train_x), batch_size= 3)
    # np.save(npy_save_main_train_predict_y, mainpipe_train_x)
    # del mainpipe_train_x
    # gc.collect()
    # mainpipe_valid_x = model_segmentation.predict(np.load(npy_valid_x), batch_size=3)
    # np.save(npy_save_main_valid_predict_y, mainpipe_valid_x)
    # del mainpipe_valid_x
    # gc.collect()
    # mainpipe_test_x = model_segmentation.predict(np.load(npy_test_x), batch_size=3)
    # np.save(npy_save_main_test_predict_y, mainpipe_test_x)
    # del mainpipe_test_x
    # gc.collect()

    # # 副支線生成訓練測試x資料
    # # dist_data 儲存
    # print("Dist_data preocessing")
    # model_judge = classification.Alexnet(input_shape=input_shape, output_class=2)
    # model_judge.compile(optimizer= Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # model_judge.load_weights(sub_judge_weights)
    #
    # dist_data = model_judge.predict(np.load(npy_train_x), batch_size=10)
    # dist_data = np.ones(dist_data[:, 0].shape) * (dist_data[:, 0] > 0.5)
    # np.save(npy_save_sub_train_dist, dist_data)
    #
    # dist_data = model_judge.predict(np.load(npy_valid_x), batch_size=10)
    # dist_data = np.ones(dist_data[:, 0].shape) * (dist_data[:, 0] > 0.5)
    # np.save(npy_save_sub_valid_dist, dist_data)
    #
    # dist_data = model_judge.predict(np.load(npy_test_x), batch_size=10)
    # dist_data = np.ones(dist_data[:, 0].shape) * (dist_data[:, 0] > 0.5)
    # np.save(npy_save_sub_test_dist, dist_data)
    # del dist_data, model_judge
    # gc.collect()

    # # subpipe 預測
    # print("Subpipe processing training")
    # dist_data = np.load(npy_save_sub_train_dist)
    #
    # high_x = np.delete((np.load(npy_train_x)), np.where(dist_data == 0), axis=0)
    # high_y = np.delete((np.load(npy_train_y)), np.where(dist_data == 0), axis=0)
    # model_segmentation.load_weights(sub_segHigh_weights)
    # predict_high_y = model_segmentation.predict(high_x, batch_size=3)
    # np.save(npy_save_sub_train_high_predict_y, predict_high_y)
    # del high_x, high_y, predict_high_y
    # gc.collect()

    # low_x = np.delete(np.load(npy_train_x), np.where(dist_data == 1), axis=0)
    # low_y = np.delete(np.load(npy_train_y), np.where(dist_data == 1), axis=0)
    # model_segmentation.load_weights(sub_segLow_weights)
    # predict_low_y = model_segmentation.predict(low_x, batch_size=3)
    # np.save(npy_save_sub_train_low_predict_y, predict_low_y)
    # del low_x, low_y, predict_low_y
    # gc.collect()

    # subpipe 預測
    # print("Subpipe processing valid.")
    # dist_data = np.load(npy_save_sub_valid_dist)
    #
    # high_x = np.delete((np.load(npy_valid_x)), np.where(dist_data == 0), axis=0)
    # high_y = np.delete((np.load(npy_valid_y)), np.where(dist_data == 0), axis=0)
    # model_segmentation.load_weights(sub_segHigh_weights)
    # predict_high_y = model_segmentation.predict(high_x, batch_size=3)
    # np.save(npy_save_sub_valid_high_predict_y, predict_high_y)
    # del high_x, high_y, predict_high_y
    # gc.collect()
    #
    # low_x = np.delete(np.load(npy_valid_x), np.where(dist_data == 1), axis=0)
    # low_y = np.delete(np.load(npy_valid_y), np.where(dist_data == 1), axis=0)
    # model_segmentation.load_weights(sub_segLow_weights)
    # predict_low_y = model_segmentation.predict(low_x, batch_size=3)
    # np.save(npy_save_sub_valid_low_predict_y, predict_low_y)
    # del low_x, low_y, predict_low_y
    # gc.collect()

    # print("Subpipe processing test.")
    # dist_data = np.load(npy_save_sub_test_dist)
    #
    # high_x = np.delete((np.load(npy_test_x)), np.where(dist_data == 0), axis=0)
    # high_y = np.delete((np.load(npy_test_y)), np.where(dist_data == 0), axis=0)
    # model_segmentation.load_weights(sub_segHigh_weights)
    # predict_high_y = model_segmentation.predict(high_x, batch_size=3)
    # np.save(npy_save_sub_test_high_predict_y, predict_high_y)
    # del high_x, high_y, predict_high_y
    # gc.collect()
    #
    # low_x = np.delete(np.load(npy_test_x), np.where(dist_data == 1), axis=0)
    # low_y = np.delete(np.load(npy_test_y), np.where(dist_data == 1), axis=0)
    # model_segmentation.load_weights(sub_segLow_weights)
    # predict_low_y = model_segmentation.predict(low_x, batch_size=3)
    # np.save(npy_save_sub_test_low_predict_y, predict_low_y)
    # del low_x, low_y, predict_low_y
    # gc.collect()

    # print("Subpipe train high and low predict concate.")
    # predict_high_y = np.load(npy_save_sub_train_high_predict_y)
    # predict_low_y = np.load(npy_save_sub_train_low_predict_y)
    # dist_data = np.load(npy_save_sub_train_dist)
    # high_order = dist_data == 1
    # index_high = 0
    # index_low = 0
    # predict_y = []
    # print("Concate predict data")
    # for index in range(len(high_order)):
    #     if high_order[index]:
    #         predict_y.append(predict_high_y[index_high])
    #         # predict_y[index] = predict_high_y[index_high]
    #         index_high += 1
    #         # np.delete(high_y_predict, 0, axis=0)
    #     else:
    #         predict_y.append(predict_low_y[index_low])
    #         # predict_y[index] = predict_low_y[index_low]
    #         index_low += 1
    #         # np.delete(low_y_predict, 0, axis=0)
    # predict_y = np.array(predict_y)
    # np.save(npy_save_sub_train_predict_y, predict_y)
    # ooutput_y = postprocessing.check_threshold(predict_y,
    #                                            size=(256, 256, 1),
    #                                            threshold=0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_outputY.npy", ooutput_y)

    # print("Subpipe valid high and low predict concate.")
    # predict_high_y = np.load(npy_save_sub_valid_high_predict_y)
    # predict_low_y = np.load(npy_save_sub_valid_low_predict_y)
    # dist_data = np.load(npy_save_sub_valid_dist)
    # high_order = dist_data == 1
    # index_high = 0
    # index_low = 0
    # predict_y = []
    # print("Concate predict data")
    # for index in range(len(high_order)):
    #     if high_order[index]:
    #         predict_y.append(predict_high_y[index_high])
    #         # predict_y[index] = predict_high_y[index_high]
    #         index_high += 1
    #         # np.delete(high_y_predict, 0, axis=0)
    #     else:
    #         predict_y.append(predict_low_y[index_low])
    #         # predict_y[index] = predict_low_y[index_low]
    #         index_low += 1
    #         # np.delete(low_y_predict, 0, axis=0)
    # predict_y = np.array(predict_y)
    # np.save(npy_save_sub_valid_predict_y, predict_y)
    # ooutput_y = postprocessing.check_threshold(predict_y,
    #                                            size=(256, 256, 1),
    #                                            threshold=0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_outputY.npy", ooutput_y)

    # print("Subpipe test high and low predict concate.")
    # predict_high_y = np.load(npy_save_sub_test_high_predict_y)
    # predict_low_y = np.load(npy_save_sub_test_low_predict_y)
    # dist_data = np.load(npy_save_sub_test_dist)
    # high_order = dist_data == 1
    # index_high = 0
    # index_low = 0
    # predict_y = []
    # print("Concate predict data")
    # for index in range(len(high_order)):
    #     if high_order[index]:
    #         predict_y.append(predict_high_y[index_high])
    #         # predict_y[index] = predict_high_y[index_high]
    #         index_high += 1
    #         # np.delete(high_y_predict, 0, axis=0)
    #     else:
    #         predict_y.append(predict_low_y[index_low])
    #         # predict_y[index] = predict_low_y[index_low]
    #         index_low += 1
    #         # np.delete(low_y_predict, 0, axis=0)
    # predict_y = np.array(predict_y)
    # np.save(npy_save_sub_test_predict_y, predict_y)
    # ooutput_y = postprocessing.check_threshold(predict_y,
    #                                            size=(256, 256, 1),
    #                                            threshold=0.5)
    # np.save(".\\npy\\20201118_256_51984_UNet(2RDB8-DtoU-5)_CE-predictTrainData(V1)-subpipe_outputY.npy", ooutput_y)
    # del predict_y, predict_high_y, predict_low_y, model_judge, model_segmentation
    # gc.collect()

    for i in range(len(name)):
        print("Building model.")
        # model_select = model.UNet_DtoU5(block=model.RDBlocks,
        #                                 name="unet_2RD-5",
        #                                 input_size=input_shape,
        #                                 block_num=2)
        if i <= 1:
            model_select = model.Fusion_net(activation= activation[i], size=(256, 256, 1))
            model_select.load_weights(FusionNet_model_path[i])
        else:
            continue
            model_select = model.Fusion_net_twoActivation(size=(256, 256, 1))

        epochs = 50
        batch = 10
        train_flag = 0
        test_flag = 1

        print("Compile model.")
        model_select.compile(optimizer= Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

        print("Model building.")
        model_build = model.model(model=model_select,
                                  name=name[i],
                                  size=input_shape)

        print("Select train model：{}".format(model_build.name))

        if train_flag:
            print("Loading data.")
            mainpipe_train_x = np.load(npy_save_main_train_predict_y)
            subpipe_train_x = np.load(npy_save_sub_train_predict_y)
            train_y = np.load(npy_train_y)
            mainpipe_valid_x = np.load(npy_save_main_valid_predict_y)
            subpipe_valid_x = np.load(npy_save_sub_valid_predict_y)
            valid_y = np.load(npy_valid_y)
            print("End loading data.")
            print("Start training.")
            model_build.train(x_train=[mainpipe_train_x, subpipe_train_x], y_train=train_y,
                              x_valid= [mainpipe_valid_x, subpipe_valid_x], y_valid= valid_y,
                              batch_size=batch, epochs= epochs)
            print("End training.")

        if test_flag:
            print("Loading data.")
            mainpipe_test_x = np.load(npy_save_main_test_predict_y)
            subpipe_test_x = np.load(npy_save_sub_test_predict_y)
            test_y = np.load(npy_test_y)
            print("End loading data.")
            print("Start testing.")
            model_build.test(x_test= [mainpipe_test_x, subpipe_test_x], y_test=test_y, data_start= test_data_start,batch_size=batch)
            print("End testing.")