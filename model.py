import numpy as np
from keras.models import *
from keras.layers import *
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras import backend as keras
import keras
import time
import tensorflow as tf
import model_deeplab3plus as DV3


# AdaBooster-CNN
# from sklearn.preprocessing import LabelBinarizer, Binarizer
from copy import deepcopy
from numpy.core.umath_tests import inner1d
# AdaBooster-CNN  END

# segnet
from layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
# from matplotlib import pyplot as plt
import excel
import estimate
import cv2
import postprocessing
import os
import data
import loss
from lr_range import callbacks_list
import rasterio

class model():
    def __init__(self, model, name, size= (256,256,4)):
        self.heights= size[0]
        self.widths= size[1]
        self.channels= size[2]
        self.shape = size
        self.name = name
        self.model = model

    def train(self, x_train, y_train, epochs= 100, batch_size= 10, validation_ratio= 0.125):
    # def train_self_validation(self, x_train, y_train, valid_x, valid_y, epochs=100, batch_size=10):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # if x_train.ndim == 3:
        #     x_train = np.expand_dims(x_train, axis=-1)
        # if y_train.ndim == 3:
        #     y_train = np.expand_dims(y_train, axis=-1)

        history = self.model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[saveModel, checkBestPoint, reduce_lr])
        # history = self.model.fit(x_train, y_train, validation_data= (valid_x, valid_y), batch_size=batch_size,
        #                          epochs=epochs,
        #                          callbacks=[saveModel, checkBestPoint, reduce_lr])
                                 # validation_split=validation_ratio,


        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()
    def train_generator(self, dataset, train_num, valid_num, epochs=100, batch_size=10):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

        history = self.model.fit_generator(self.myGenerator(start_num= 1, batch_size= batch_size, dataset= dataset, total_num= train_num),
                                           samples_per_epoch=train_num // batch_size * batch_size,
                                           nb_epoch=epochs,
                                           validation_data=self.myGenerator(start_num= 1 + train_num, batch_size= batch_size, dataset= dataset, total_num= valid_num),
                                           nb_val_samples=(valid_num // batch_size * batch_size),
                                           verbose=1,
                                           callbacks=[saveModel, checkBestPoint, reduce_lr])

        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()
    def train_withSanpleWeights(self, x_train, y_train, sample_weight, epochs= 100, batch_size= 10, validation_ratio= 0.125):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # if x_train.ndim == 3:
        #     x_train = np.expand_dims(x_train, axis=-1)
        # if y_train.ndim == 3:
        #     y_train = np.expand_dims(y_train, axis=-1)

        history = self.model.fit(x_train, y_train,sample_weight=sample_weight, batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[saveModel, checkBestPoint, reduce_lr])
        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        # print(history.history.keys())
        # fig = plt.figure()
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')

        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='lower left')
        # fig.savefig('.\\result\performance\\' + self.name + '.png')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()
    def train_sampleWeights(self, x_train, y_train,batch_epoch,
                            epochs= 100,
                            batch_size= 10,
                            validation_ratio= 0.125,
                            threshold = 0.5):
        def update_sample_weights(x, y, model, sample_weight, learning_rate = 2, threshold = 0.5):
            #         CNN 訓練資料預測
            y_pred = model.predict(x)
            y_predict_proba = y_pred
            # repalce zero
            y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

            # 第二版 使用原本的概念
            n_classes = 2

            y_mse = np.mean((y_predict_proba - y) ** 2, axis=(1, 2, 3))
            y_far = (y_mse > threshold) * np.ones(y_mse.shape)
            y_closed = (y_mse <= threshold) * np.ones(y_mse.shape)
            y_refresh = y_far * (-1. / (n_classes - 1)) * y_mse + y_closed * y_mse

            # for sample weight update
            intermediate_variable = (
                -1. * learning_rate * (((n_classes - 1) / n_classes) * y_refresh))
            # intermediate_variable = (-1. * self.learning_rate_ * y_refresh)


            # update sample weight
            sample_weight *= np.exp(intermediate_variable)
            # normalize sample weight
            sample_weight /= np.sum(sample_weight, axis=0)
            return sample_weight


        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # if x_train.ndim == 3:
        #     x_train = np.expand_dims(x_train, axis=-1)
        # if y_train.ndim == 3:
        #     y_train = np.expand_dims(y_train, axis=-1)

        sample_weights = np.ones(len(x_train)) / len(x_train)
        for i in range(epochs // batch_epoch):
            weight_path = ".\\result\\model_record\\" + self.name
            checkBestPoint = ModelCheckpoint(
                weight_path + '-bestweights-' + str(i+1) + 'epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
                monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
            saveModel = ModelCheckpoint(
                weight_path + '-weights-' + str(i+1) + 'epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1,
                save_best_only=False, save_weights_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

            history = self.model.fit(x_train, y_train,
                                     sample_weight=sample_weights,
                                     batch_size= batch_size,
                                     epochs=batch_epoch,
                                     validation_split=validation_ratio,
                                     callbacks=[saveModel,checkBestPoint, reduce_lr])
            sample_weights = update_sample_weights(x_train, y_train, self.model, sample_weights, threshold = threshold)

            ex_loss = excel.Excel()
            ex_loss.write_excel("c2", "sample_weights")
            ex_loss.write_excel("c3", sample_weights, vertical= True)
            ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
            ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_sampleweight_loss"+str(i+1)+".xlsx")
            ex_loss.close_excel()

        self.model.save(".\\result\model_record\\" + self.name + '.h5')

    def test(self, x_test, y_test, data_start, batch_size= 10, save_path = None):
        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)
        if save_path == None:
            save_path = self.name

        y_predict = self.model.predict(x_test, batch_size=batch_size)

        print("Postprocessing.")
        self.postprocessing(test_y= y_test, predict_y= y_predict, save_path= save_path, data_start= data_start)

    def postprocessing(self, test_y, predict_y, save_path, data_start, threshold = 0.5):
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

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(predict_y,
                                                  size=(self.heights, self.widths, 1),
                                                  threshold= threshold)

        print("Estimate.")
        iou = estimate.IOU(test_y, y_output, self.widths, len(test_y))
        (precision, recall, F1) = estimate.F1_estimate(test_y, y_output, self.widths, len(test_y))
        avr_iou = np.sum(iou) / len(test_y)
        avr_precision = np.sum(precision) / len(test_y)
        avr_recall = np.sum(recall) / len(test_y)
        avr_F1 = np.sum(F1) / len(test_y)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(test_y)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("a2", "image num")
        for index in range(len(test_y)):
            ex_iou.write_excel("a" + str(index + 3), str(data_start + index))
        ex_iou.write_excel("f1", "precision", vertical=True)
        ex_iou.write_excel("f2", precision, vertical=True)
        ex_iou.write_excel("g1", "avr_precision", vertical=True)
        ex_iou.write_excel("g2", avr_precision, vertical=True)
        ex_iou.write_excel("h1", "recall", vertical=True)
        ex_iou.write_excel("h2", recall, vertical=True)
        ex_iou.write_excel("i1", "avr_recall", vertical=True)
        ex_iou.write_excel("i2", avr_recall, vertical=True)
        ex_iou.write_excel("j1", "F1", vertical=True)
        ex_iou.write_excel("j2", F1, vertical=True)
        ex_iou.write_excel("k1", "avr_F1", vertical=True)
        ex_iou.write_excel("k2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def test_ada(self, x_test, y_test, data_start, batch_size= 10, threshold= 0.5, save_path = None):
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

        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)
        if save_path == None:
            save_path = self.name
        file_name_head = "20200703_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE"
        file_name = np.array([file_name_head + "_0.h5",
                              file_name_head + "_1.h5",
                              file_name_head + "_2.h5",
                              file_name_head + "_3.h5",
                              file_name_head + "_4.h5",
                              file_name_head + "_5.h5",
                              file_name_head + "_6.h5",
                              file_name_head + "_7.h5",
                              file_name_head + "_8.h5",
                              file_name_head + "_9.h5"])

        y_predict = np.zeros(y_test.shape, dtype= np.float32)
        for model_file in file_name:
            print("Read model weight {}".format(".\\result\\model_record\\V3.2_test\\" + model_file))
            self.model.load_weights(".\\result\\model_record\\V3.2_test\\" + model_file)
            result = self.model.predict(x_test, batch_size=batch_size) / len(file_name)
            print("Result = {}".format(np.sum(result)))
            y_predict = y_predict + result

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size=(self.heights, self.widths, 1),
                                                  threshold= threshold)
        # y_output = postprocessing.recover_area(y_output,
        #                                        start=data_start,
        #                                        number=len(y_test),
        #                                        threshold=land_thre)

        # # test the locate
        #
        # kernel = np.ones((10, 10), np.uint8)
        # accurate = np.zeros(len(x_test), dtype=np.uint8)
        # gt_true = np.zeros(len(x_test), dtype=np.uint8)
        # y_range = np.zeros((len(x_test), self.heights, self.widths), dtype=np.uint8)
        # for i in range(len(x_test)):
        #     # 侵蝕
        #     y_erode = cv2.erode(y_output[i], kernel)
        #     # 膨脹
        #     y_dilate = cv2.dilate(y_erode, kernel)
        #
        #     gt_true[i] = 1 if np.max(y_test[i])==1 else 0
        #
        #     # 找出最可能的地方
        #     y_range[i] = np.dot(y_dilate,y_predict[i])[:,:,0]
        #     if np.max(y_range[i]) == 0:
        #         accurate[i] = 0
        #         continue
        #
        #     ((x,y)) = np.where(y_range[i] == np.max(y_range[i]))
        #     # 評估
        #     index = len(x) // 2
        #     if y_test[i, x[0], y[0]] == 1:
        #         accurate[i]=1
        #
        # # test the locate end

        print("Estimate.")
        iou = estimate.IOU(y_test, y_output, self.widths, len(y_test))
        (precision, recall, F1) = estimate.F1_estimate(y_test, y_output, self.widths, len(y_test))
        avr_iou = np.sum(iou) / len(y_test)
        avr_precision = np.sum(precision) / len(y_test)
        avr_recall = np.sum(recall) / len(y_test)
        avr_F1 = np.sum(F1) / len(y_test)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(y_test)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("a2", "image num")
        for index in range(len(y_test)):
            ex_iou.write_excel("a" + str(index + 3), str(data_start + index))
        ex_iou.write_excel("f1", "precision", vertical=True)
        ex_iou.write_excel("f2", precision, vertical=True)
        ex_iou.write_excel("g1", "avr_precision", vertical=True)
        ex_iou.write_excel("g2", avr_precision, vertical=True)
        ex_iou.write_excel("h1", "recall", vertical=True)
        ex_iou.write_excel("h2", recall, vertical=True)
        ex_iou.write_excel("i1", "avr_recall", vertical=True)
        ex_iou.write_excel("i2", avr_recall, vertical=True)
        ex_iou.write_excel("j1", "F1", vertical=True)
        ex_iou.write_excel("j2", F1, vertical=True)
        ex_iou.write_excel("k1", "avr_F1", vertical=True)
        ex_iou.write_excel("k2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def myGenerator(self, start_num, batch_size, dataset, total_num):
        x = np.zeros([batch_size, 256, 256, 4], dtype= np.float32)
        y = np.zeros([batch_size, 256, 256, 1], dtype= np.float32)
        load_path = "E:\\allen\\data\\dataset\\"
        while True:
            for i in range(total_num // batch_size):
                for index in range(start_num + i*batch_size, start_num + (i+1)*batch_size):
                    data_raster = rasterio.open(load_path + dataset + '\\x\\' + str(index) + '.tif')
                    data_raster_img = data_raster.read().transpose((1, 2, 0))
                    x[(index - (start_num + i*batch_size))] = (data_raster_img.astype('float32') / np.max(data_raster_img))

                for index in range(start_num + i*batch_size, start_num + (i+1)*batch_size):
                    data_raster = rasterio.open(load_path + dataset + '\\y\\' + str(index) + '.tif')
                    data_raster_nr = data_raster.read(1)
                    y[index - (start_num + i*batch_size)] = np.expand_dims(np.ones(data_raster_nr.shape, dtype=np.float32) * (data_raster_nr > 0), axis=-1)
                yield (x,y)
                # yield ({'input_1': x}, {'output': y})

class AdaBoostCNN(object):
    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'epochs', 'input_size', 'output_size', 'name', 'batch_size']
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise ValueError(keyword + ":  Wrong keyword used --- check spelling")

        n_estimators = 10
        learning_rate = 2
        #### CNN (5)
        epochs = 6
        input_size = (256, 256, 4)
        output_size = (256, 256, 1)

        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            ### CNN:
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')
            if 'input_size' in kwargs: input_size = kwargs.pop('input_size')
            if 'output_size' in kwargs: output_size = kwargs.pop('output_size')

            if 'batch_size' in kwargs: batch_size = kwargs.pop('batch_size')
            if 'name' in kwargs: name = kwargs.pop('name')

        self.name = name
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate_ = learning_rate
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.ones(self.n_estimators)

        self.epochs = epochs
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

    def fit(self, x, y):
        self.n_samples = x.shape[0]
        self. classes_ = np.array(sorted(list(set(y.flatten()))))

#         訓練迴圈
        for estimator_index in range(self.n_estimators):
            print("No. {} training process.".format(estimator_index + 1))
            if estimator_index == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples

            sample_weight, estimator_weight, estimator_error= self.boost(x, y, sample_weight, order = estimator_index)

            self.estimator_errors_[estimator_index] = estimator_error
            self.estimator_weights_[estimator_index] = estimator_weight

        print("sample weight={}".format(sample_weight))
        print("estimatore weight={}".format(self.estimator_weights_))
        print("estimator error={}".format(self.estimator_errors_))
        return self

    def boost(self, x, y, sample_weight, order, threshold = 0.5):


        if order == 0:
            estimator = self.base_estimator
        else:
            estimator = self.estimator

        # CNN 訓練
        weight_path = ".\\result\model_record\\" + self.name + '_' + str(order)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
                                    verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto') # 7/9 新增
        estimator.fit(x, y, sample_weight= sample_weight, epochs= self.epochs, batch_size= self.batch_size, validation_split=0.125, callbacks=[saveModel, reduce_lr])
        estimator.save(".\\result\model_record\\" + self.name + '_' + str(order) + '.h5')
#         CNN 訓練資料預測
        y_pred = estimator.predict(x)
#         CNN 檢查預測結果比較
        y_threshold = y_pred >= threshold
        y_threshold = np.ones(y_threshold.shape, dtype=np.uint8) * y_threshold
        y_incorrect = y_threshold != y
        incorrect = np.sum(y_incorrect, axis=(1, 2, 3)) / self.input_size[0] ** 2
        #
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

        # y_predict_proba = estimator.predict_proba(x)
        y_predict_proba = y_pred

        # repalce zero
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        # -------------------------以上完成-----------------------------
        self.n_classes_ = 2
        self.classes_ = list([0, 1])

        # 第一次正常運作 第一版
        # y_mse = np.mean((y_predict_proba - y) ** 2, axis=(1, 2, 3)) / self.output_size[0]**2
        # y_mse = np.mean((y_predict_proba - y) ** 2, axis=(1, 2, 3)) # 7/7
        # print("y_mse.shape = {}.".format(y_mse.shape))
        # print("y_mse.max() = {}.".format(y_mse.max()))
        # print("y_mse.min() = {}.".format(y_mse.min()))
        # y_mse = 1 - y_mse # 7/3
        # y_refresh = y_mse

        # 第二版 使用原本的概念
        y_mse = np.mean((y_predict_proba - y) ** 2, axis=(1, 2, 3))
        y_far = (y_mse > threshold) * np.ones(y_mse.shape)
        y_closed = (y_mse <= threshold) * np.ones(y_mse.shape)
        y_refresh = y_far * (-1. / (self.n_classes_ - 1)) * y_mse + y_closed * y_mse

        # for sample weight update
        intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) * y_refresh))
        # intermediate_variable = (-1. * self.learning_rate_ * y_refresh)

        # dot iterate for each row
        # ---------------------以上不懂------------------------------------------
        # update sample weight
        sample_weight *= np.exp(intermediate_variable)
        print("sample weight.max = {}".format(sample_weight.max()))
        # sample_weight = 1 - sample_weight 7/2
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            print("sample weight sum <= 0")
            return None, None, None
        # normalize sample weight
        sample_weight /= sample_weight_sum

        self.estimator = estimator
        return sample_weight, 1, estimator_error

    def test(self, model_weight_order, x_test, y_test, data_start, batch_size= 10, threshold= 0.5, save_path = None):
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

        model_weights_path = ".\\result\model_record\\" + self.name + '_' + str(model_weight_order) + '.h5'

        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)
        if save_path == None:
            save_path = self.name

        self.base_estimator.load_weights(model_weights_path)
        y_predict = self.base_estimator.predict(x_test, batch_size=batch_size)

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size= self.output_size,
                                                  threshold= threshold)

        print("Estimate.")
        iou = estimate.IOU(y_test, y_output, self.output_size[0], len(y_test))
        (precision, recall, F1) = estimate.F1_estimate(y_test, y_output, self.output_size[0], len(y_test))
        avr_iou = np.sum(iou) / len(y_test)
        avr_precision = np.sum(precision) / len(y_test)
        avr_recall = np.sum(recall) / len(y_test)
        avr_F1 = np.sum(F1) / len(y_test)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(y_test)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("a2", "image num")
        for index in range(len(y_test)):
            ex_iou.write_excel("a" + str(index + 3), str(data_start + index))
        ex_iou.write_excel("f1", "precision", vertical=True)
        ex_iou.write_excel("f2", precision, vertical=True)
        ex_iou.write_excel("g1", "avr_precision", vertical=True)
        ex_iou.write_excel("g2", avr_precision, vertical=True)
        ex_iou.write_excel("h1", "recall", vertical=True)
        ex_iou.write_excel("h2", recall, vertical=True)
        ex_iou.write_excel("i1", "avr_recall", vertical=True)
        ex_iou.write_excel("i2", avr_recall, vertical=True)
        ex_iou.write_excel("j1", "F1", vertical=True)
        ex_iou.write_excel("j2", F1, vertical=True)
        ex_iou.write_excel("k1", "avr_F1", vertical=True)
        ex_iou.write_excel("k2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def predict_proba(self, X):
        proba = sum(self._samme_proba(estimator, self.n_classes_, X) for estimator in self.estimators_)

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (self.n_classes_ - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba
    def _samme_proba(self, estimator, n_classes, X):
        proba = estimator.predict_proba(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1. / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

class Judgement_model(object):
    def __init__(self, model, name, input_shape= (256,256,4), classes = 2):
        self.heights= input_shape[0]
        self.widths= input_shape[1]
        self.channels= input_shape[2]
        self.shape = input_shape
        self.classes = classes
        self.name = name
        self.model = model

    def GT_data_transfer(self, excel_file, total_num, extract_index= "iou", threshold= 0.9):
        file = excel.Excel(file_path=excel_file)
        print("Excel file {} is opened.".format(excel_file))

        if extract_index == "iou" or extract_index == "IoU":
            index = file.read_excel(start="c3:c" + str(2 + total_num))
        elif extract_index == "precision" or extract_index == "Precision":
            index = file.read_excel(start="e2:e" + str(1 + total_num))
        elif extract_index == "recall" or extract_index == "Recall":
            index = file.read_excel(start="g2:g" + str(1 + total_num))
        elif extract_index == "f1" or extract_index == "F1":
            index = file.read_excel(start="i2:i" + str(1 + total_num))
        else:
            print("extrac_index：{}. ERROR".format(extract_index))
            raise ValueError
        file.close_excel()

        index = np.array(index, dtype=np.float32)

        print("Index is {}.".format(extract_index))
        print("Index shape {}".format(index.shape))
        print("Index dtype {}".format(index.dtype))
        low_result = np.ones(index.shape) * (index <= threshold)
        high_result = np.ones(index.shape) * (index > threshold)

        output_y = np.zeros((len(index), 2))
        output_y[:, 0] = high_result
        output_y[:, 1] = low_result

        return output_y

    def train(self, x, y, epochs= 100, batch_size= 10, validation_ratio= 0.125):
        weight_path = ".\\result\\model_record\\" + self.name
        checkBestPoint = ModelCheckpoint(weight_path + '-bestweights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        saveModel = ModelCheckpoint(weight_path + '-weights-epoch{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5', verbose=1, save_best_only=False, save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
        # stop_training = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, mode="min")

        # if x_train.ndim == 3:
        #     x_train = np.expand_dims(x_train, axis=-1)
        # if y_train.ndim == 3:
        #     y_train = np.expand_dims(y_train, axis=-1)

        # history = self.model.fit(x_train, y_train, batch_size= batch_size, epochs=epochs, validation_split=validation_ratio, callbacks=[saveModel, checkBestPoint, callbacks_list[0], callbacks_list[1]])
        history = self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                                 validation_split=validation_ratio,
                                 callbacks=[saveModel, checkBestPoint, reduce_lr])

        self.model.save(".\\result\model_record\\" + self.name + '.h5')

        ex_loss = excel.Excel()
        ex_loss.write_loss_and_iou(self.name, history.history['loss'], history.history['val_loss'], 0, 0)
        ex_loss.save_excel(file_name=".\\result\data\\" + self.name + "_loss.xlsx")
        ex_loss.close_excel()

    def test(self, x, y, data_start, batch_size= 10, save_path = None):
        if save_path == None:
            save_path = self.name

        predict_y = self.model.predict(x, batch_size=batch_size)

        print("Postprocessing.")
        self.postprocessing(test_y= y, predict_y= predict_y, save_path= save_path, data_start= data_start)

    def postprocessing(self, test_y, predict_y, save_path, data_start, threshold = 0.5):
        print("Check the threshold.")
        output_y = np.ones(predict_y[:, 0].shape) * (predict_y[:, 0] > threshold)

        print("Estimate.")
        correct_y = np.ones(output_y.shape) * (output_y == test_y[:, 0])
        acc = np.sum(correct_y) / len(test_y)
        print("Average Acc：{}".format(acc))

        ex_acc = excel.Excel()
        ex_acc.write_excel("a1", save_path)
        ex_acc.write_excel("a2", "Order")
        for index in range(len(test_y)):
            ex_acc.write_excel("a" + str(index + 3), str(data_start + index))
        ex_acc.write_excel("b2", "Value of GT")
        ex_acc.write_excel("b3", test_y[:, 0], vertical= True)
        ex_acc.write_excel("d2", "Value of prediction")
        ex_acc.write_excel("d3", output_y, vertical=True)
        ex_acc.write_excel("f2", "Value of correct")
        ex_acc.write_excel("f3", correct_y, vertical=True)
        ex_acc.write_excel("h2", "Value of Acc")
        ex_acc.write_excel("h3", acc)

        ex_acc.save_excel(file_name=".\\result\data\\" + save_path + "_acc.xlsx")
        ex_acc.close_excel()
        print(correct_y.shape)
        print(output_y.shape)
        print(test_y[:, 0].shape)
        print(acc.shape)

'''
    model：給所有的模型架構
                [0] 為專門模型架構
                [1] 為評判模型架構
'''
class SI_mark_processing(object):
    def __init__(self, model, name, input_shape= (256,256,4), pipelines = 2):
        self.heights= input_shape[0]
        self.widths= input_shape[1]
        self.channels= input_shape[2]
        self.shape = input_shape
        self.pipelines = pipelines
        self.name = name
        self.pipe_model_struct = model[0]
        self.judgement_model_struct = model[1]

    def predict(self, x, y, judge_weights_path, pipe_weights_path, save_path, data_start):
        if len(pipe_weights_path) != self.pipelines:
            print("Pipelines num is different.")
            raise ValueError

        print("Judge processing.")
        dist_data = self.judge(x, judge_weights_path)
        print("Finish judge processing.")

        print("Data distribution.")
        high_x = x
        high_y = y
        high_x = np.delete(high_x, np.where(dist_data[:, 1] == 1), axis=0)
        high_y = np.delete(high_y, np.where(dist_data[:, 1] == 1), axis=0)
        low_x = x
        low_y = y
        low_x = np.delete(low_x, np.where(dist_data[:, 0] == 1), axis=0)
        low_y = np.delete(low_y, np.where(dist_data[:, 0] == 1), axis=0)
        print("high_x shape:{}\thigh_y shape:{}".format(high_x.shape, high_y.shape))
        print("low_x shape:{}\tlow_y shape:{}".format(low_x.shape, low_y.shape))
        print("End data distribution.")

        predict_y = np.zeros(y.shape)
        print("Mark processing.")
        print("Predict high data pipelines")
        self.pipe_model_struct.load_weights(pipe_weights_path[0])
        predict_high_y = self.pipe_model_struct.predict(high_x, batch_size=3)
        print("End predict high data pipelines")

        print("Predict low data pipelines")
        self.pipe_model_struct.load_weights(pipe_weights_path[1])
        predict_low_y = self.pipe_model_struct.predict(low_x, batch_size=3)
        print("End predict low data pipelines")

        high_order = dist_data[:, 0] == 1
        index_high = 0
        index_low = 0
        print("Concate predict data")
        for index in range(len(y)):
            if high_order[index]:
                predict_y[index] = predict_high_y[index_high]
                index_high += 1
                # np.delete(high_y_predict, 0, axis=0)
            else:
                predict_y[index] = predict_low_y[index_low]
                index_low += 1
                # np.delete(low_y_predict, 0, axis=0)
        print("End concate predict data")
        print("Postprocessing")
        # self.postprocessing(y, predict_y, save_path, data_start, threshold = 0.5)
        print("End postprocessing")
        print("End mark processing.")

        return predict_y

    def judge(self, x, judge_weights_path, batch_size = 10, threshold = 0.5):
        self.judgement_model_struct.load_weights(judge_weights_path)
        predict_y = self.judgement_model_struct.predict(x, batch_size=batch_size)
        output_y = np.ones(predict_y.shape) * (predict_y > threshold)
        # output_y[0]：高於門檻值
        # output_y[1]：低於等於門檻值
        return output_y

    def postprocessing(self, test_y, predict_y, save_path, data_start, threshold = 0.5):
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

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(predict_y,
                                                  size=(self.heights, self.widths, 1),
                                                  threshold= threshold)

        print("Estimate.")
        iou = estimate.IOU(test_y, y_output, self.widths, len(test_y))
        (precision, recall, F1) = estimate.F1_estimate(test_y, y_output, self.widths, len(test_y))
        avr_iou = np.sum(iou) / len(test_y)
        avr_precision = np.sum(precision) / len(test_y)
        avr_recall = np.sum(recall) / len(test_y)
        avr_F1 = np.sum(F1) / len(test_y)
        print("Average IOU：{}".format(avr_iou))

        print('Save the result.')
        mkdir(".\\result\image\\" + save_path)
        for index in range(len(test_y)):
            img_save = y_output[index] * 255
            cv2.imwrite(".\\result\image\\" + save_path + '\\{}.png'.format(data_start + index), img_save)
            print('Save image:{}'.format(data_start + index))

        ex_iou = excel.Excel()
        ex_iou.write_loss_and_iou(save_path, 0, 0, iou, avr_iou)
        ex_iou.write_excel("a2", "image num")
        for index in range(len(test_y)):
            ex_iou.write_excel("a" + str(index + 3), str(data_start + index))
        ex_iou.write_excel("f1", "precision", vertical=True)
        ex_iou.write_excel("f2", precision, vertical=True)
        ex_iou.write_excel("g1", "avr_precision", vertical=True)
        ex_iou.write_excel("g2", avr_precision, vertical=True)
        ex_iou.write_excel("h1", "recall", vertical=True)
        ex_iou.write_excel("h2", recall, vertical=True)
        ex_iou.write_excel("i1", "avr_recall", vertical=True)
        ex_iou.write_excel("i2", avr_recall, vertical=True)
        ex_iou.write_excel("j1", "F1", vertical=True)
        ex_iou.write_excel("j2", F1, vertical=True)
        ex_iou.write_excel("k1", "avr_F1", vertical=True)
        ex_iou.write_excel("k2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

def Fusion_net(activation, size = (256, 256, 1)):
    # first input
    input_1 = Input(size)
    conv_11 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input_1)
    conv_12 = Conv2D(64, 3, activation='relu', padding='same')(conv_11)

    # second input
    input_2 = Input(size)
    conv_21 = Conv2D(64, 3, activation='relu', padding='same')(input_2)
    conv_22 = Conv2D(64, 3, activation='relu', padding='same')(conv_21)

    # input_3 = Input(size)
    # conv_31 = Conv2D(64, 3, activation='relu', padding='same')(input_3)
    # conv_32 = Conv2D(64, 3, activation='relu', padding='same')(conv_31)
    # fusion part
    con = concatenate([conv_12, conv_22])
    # con = concatenate([con, conv_32])
    fusion_1 = Conv2D(64, 3, activation='relu', padding='same')(con)
    fusion_2 = Conv2D(64, 3, activation='relu', padding='same')(fusion_1)
    output = Conv2D(1, 1, activation=activation, padding= "same")(fusion_2)

    model = Model(inputs= [input_1, input_2], outputs= output)
    return model

def Unet(size= ( 256, 256, 4)):
    input = Input(size)
    conv1 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2,2), None, 'same')(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D((2,2), None, 'same')(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2,2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv9)

    model = Model(input = input, output = conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def Unet_elu(size= (256, 256, 5)):
    input = Input(size)
    print("input.shape : ",input.shape)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(input)
    conv1 = ELU(alpha=1.0)(conv1)
    conv1 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = ELU(alpha=1.0)(conv1)
    pool1 = MaxPooling2D((2,2), None, 'same')(conv1)

    print("pool1.shape : ",pool1.shape)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = ELU(alpha=1.0)(conv2)
    conv2 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = ELU(alpha=1.0)(conv2)
    pool2 = MaxPooling2D((2,2), None, 'same')(conv2)

    print("pool2.shape : ",pool2.shape)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = ELU(alpha=1.0)(conv3)
    conv3 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = ELU(alpha=1.0)(conv3)
    pool3 = MaxPooling2D((2,2), None, 'same')(conv3)

    print("pool3.shape : ",pool3.shape)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = ELU(alpha=1.0)(conv4)
    conv4 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = ELU(alpha=1.0)(conv4)
    drop4 = Dropout(0.5)(conv4)
    print("drop4.shape : ",drop4.shape)
    pool4 = MaxPooling2D((2,2), None, 'same')(drop4)
    print("pool4.shape : ",pool4.shape)

    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = ELU(alpha=1.0)(conv5)
    conv5 = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = ELU(alpha=1.0)(conv5)
    drop5 = Dropout(0.5)(conv5)
    print("drop5.shape : ",drop5.shape)

    up6 = Conv2D(512, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = ELU(alpha=1.0)(up6)
    print("up6.shape : ",up6.shape)
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = ELU(alpha=1.0)(conv6)
    conv6 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = ELU(alpha=1.0)(conv6)

    up7 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = ELU(alpha=1.0)(up7)
    merge7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = ELU(alpha=1.0)(conv7)
    conv7 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = ELU(alpha=1.0)(conv7)

    up8 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = ELU(alpha=1.0)(up8)
    merge8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = ELU(alpha=1.0)(conv8)
    conv8 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = ELU(alpha=1.0)(conv8)

    up9 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = ELU(alpha=1.0)(up9)
    merge9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv9 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv9 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = ELU(alpha=1.0)(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    return model

def segnet(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss.focal_loss, metrics = ['accuracy'])

    return model

def segnet_dense_inception(
        input_shape,
        n_labels,
        kernel=3,
        pool_size=(2, 2),
        output_mode="softmax"):
    def dense_block(x, filters, kernel):
        conv1 = Convolution2D(filters, (kernel, kernel), padding='same')(x)
        conv2 = Convolution2D(filters, (kernel, kernel), padding='same')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    def Inception_model(input_layer, filters):
        tower_1x1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_3x3 = Conv2D(filters, (3, 3), padding='same', activation='relu')(tower_3x3)
        tower_5x5 = Conv2D(filters, (1, 1), padding='same', activation='relu')(input_layer)
        tower_5x5 = Conv2D(filters, (5, 5), padding='same', activation='relu')(tower_5x5)
        tower_max3x3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        tower_max3x3 = Conv2D(filters, (1, 1), padding='same', activation='relu')(tower_max3x3)

        output = keras.layers.concatenate([tower_1x1, tower_3x3, tower_5x5, tower_max3x3], axis=3)
        return output

    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = dense_block(conv_1, 64, kernel)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = dense_block(pool_1, 64, kernel)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = dense_block(conv_4, 128, kernel)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = dense_block(conv_5, 256, kernel)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = dense_block(conv_5, 256, kernel)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = dense_block(conv_6, 256, kernel)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = dense_block(conv_8, 512, kernel)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = dense_block(conv_8, 512, kernel)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = dense_block(conv_9, 512, kernel)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = dense_block(pool_4, 512, kernel)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = dense_block(conv_11, 512, kernel)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = dense_block(conv_12, 512, kernel)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build enceder done..")

    # decoder
    pool_5 = Inception_model(pool_5, 128)

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = dense_block(unpool_1, 512, kernel)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = dense_block(conv_14, 512, kernel)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = dense_block(conv_15, 512, kernel)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = dense_block(unpool_2, 512, kernel)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = dense_block(conv_17, 512, kernel)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = dense_block(conv_19, 256, kernel)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = dense_block(unpool_3, 256, kernel)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = dense_block(conv_20, 256, kernel)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = dense_block(conv_22, 128, kernel)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = dense_block(unpool_4, 128, kernel)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = dense_block(conv_24, 64, kernel)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = dense_block(unpool_5, 64, kernel)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape((input_shape[0], input_shape[1], n_labels))(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")
    # model.compile(optimizer=Adam(lr=1e-4), loss=loss.focal_loss, metrics = ['accuracy'])

    return model

def RIC_Unet(size=(256,256, 5)):
    def RES_module(x, filters, kernel):
        conv1 = Convolution2D(filters, (kernel, kernel), padding='same')(x)
        conv2 = Convolution2D(filters, (kernel, kernel), padding='same')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    def DRI_module(x, f1):
        con1_1 = BatchNormalization()(x)
        con1_1 = PReLU()(con1_1)
        con1_1 = Convolution2D(f1/4, (1, 1), padding="same")(con1_1)

        con3_1 = BatchNormalization()(x)
        con3_1 = PReLU()(con3_1)
        con3_1 = Convolution2D(f1 / 4, (1, 1), padding="same")(con3_1)

        con3_3 = BatchNormalization()(x)
        con3_3 = PReLU()(con3_3)
        con3_3 = Convolution2D(f1 / 4, (1, 1), padding="same")(con3_3)
        con3_3 = BatchNormalization()(con3_3)
        con3_3 = PReLU()(con3_3)
        con3_3 = Convolution2D(f1 / 4, (3, 3), padding="same")(con3_3)
        con3_3 = keras.layers.Add()([con3_1, con3_3])

        con5_1 = BatchNormalization()(x)
        con5_1 = PReLU()(con5_1)
        con5_1 = Convolution2D(f1 / 4, (1, 1), padding="same")(con5_1)

        con5_5 = BatchNormalization()(x)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (1, 1), padding="same")(con5_5)
        con5_5 = BatchNormalization()(con5_5)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (5, 5), padding="same")(con5_5)
        con5_5 = BatchNormalization()(con5_5)
        con5_5 = PReLU()(con5_5)
        con5_5 = Convolution2D(f1 / 4, (3, 3), padding="same")(con5_5)
        con5_5 = keras.layers.Add()([con5_1, con5_5])

        max3_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        max3_3 = BatchNormalization()(max3_3)
        max3_3 = PReLU()(max3_3)
        max3_3 = Convolution2D(f1 / 4, (1, 1), padding='same')(max3_3)

        output = concatenate([con1_1, con3_3, con5_5, max3_3])
        return output

    def CAB_module(x, f):
        scale = GlobalAveragePooling2D()(x)
        scale = Dense(f, activation="relu")(scale)
        scale = Dense(f, activation="sigmoid")(scale)
        output = multiply([x, scale])
        return output


    input = Input(size)

    ri1 = Convolution2D(64, (3, 3), padding="same")(input)
    ri1 = DRI_module(ri1, 64)
    ri1 = Activation("relu")(ri1)
    ri1 = Convolution2D(128, (3, 3), strides=2, padding="same")(ri1)
    ri1 = RES_module(ri1, 128, 3)
    ri1 = Activation("relu")(ri1)

    ri2 = Convolution2D(128, (3, 3), padding="same")(ri1)
    ri2 = DRI_module(ri2, 128)
    ri2 = Activation("relu")(ri2)
    ri2 = Convolution2D(256, (3, 3), strides=2, padding="same")(ri2)
    ri2 = RES_module(ri2, 256, 3)
    ri2 = Activation("relu")(ri2)

    ri3 = Convolution2D(256, (3, 3), padding="same")(ri2)
    ri3 = DRI_module(ri3, 256)
    ri3 = Activation("relu")(ri3)
    ri3 = Convolution2D(512, (3, 3), strides=2, padding="same")(ri3)
    ri3 = RES_module(ri3, 512, 3)
    ri3 = Activation("relu")(ri3)

    ri4 = Convolution2D(512, (3, 3), padding="same")(ri3)
    ri4 = DRI_module(ri4, 512)
    ri4 = Activation("relu")(ri4)
    ri4 = Convolution2D(1024, (3, 3), strides=2, padding="same")(ri4)
    ri4 = RES_module(ri4, 1024, 3)
    ri4 = Activation("relu")(ri4)

    feature = Convolution2D(1024, (3, 3), padding="same")(ri4)
    feature = DRI_module(feature, 1024)
    feature = Activation("relu")(feature)

    dc4 = concatenate([ri4, feature])
    dc4 = Conv2DTranspose(512, (3, 3), padding="same")(dc4)
    dc4 = Activation("relu")(dc4)
    dc4 = CAB_module(dc4, 512)
    dc4 = Convolution2D(512, (3, 3), padding="same")(dc4)

    dc3 = concatenate([ri3, dc4])
    dc3 = Conv2DTranspose(512, (3, 3), padding="same")(dc3)
    dc3 = Activation("relu")(dc3)
    dc3 = CAB_module(dc3, 512)
    dc3 = Convolution2D(512, (3, 3), padding="same")(dc3)

    dc2 = concatenate([ri2, dc3])
    dc2 = Conv2DTranspose(256, (3, 3), padding="same")(dc2)
    dc2 = Activation("relu")(dc2)
    dc2 = CAB_module(dc2, 256)
    dc2 = Convolution2D(256, (3, 3), padding="same")(dc2)

    dc1 = concatenate([ri1, dc2])
    dc1 = Conv2DTranspose(128, (3, 3), padding="same")(dc1)
    dc1 = Activation("relu")(dc1)
    dc1 = CAB_module(dc1, 128)
    dc1 = Convolution2D(128, (3, 3), padding="same")(dc1)

    dc1 = Convolution2D(1, (3, 3), padding="same")(dc1)
    dc1 = BatchNormalization()(dc1)
    dc1 = Activation("sigmoid")(dc1)

    model = Model(inputs=input, outputs=dc1)
    return model

# 雙網路 model_1, model_2 需要給要訓練的兩種網路
def two_Network_weight(model_1, model_2, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]

    input_layer = Input(size, name="input_image")
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2])
    flat = Flatten()(con)
    out = Multiply()([con, Dense(2, activation="softmax")(flat)])
    print("out.shape：{}".format(out.shape))
    model_connect = Lambda(pixel_sum, output_shape=(256, 256))(out)
    model_connect = Reshape((size[0], size[1], 1))(model_connect)
    print("model_connect：{}".format(model_connect.shape))

    model = Model(inputs=input_layer, outputs=model_connect)
    return model

def two_Network_CNN(model_1, model_2, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]

    input_layer = Input(size)
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2])
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(con)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv_out = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv)

    model = Model(inputs=input_layer, outputs=conv_out)
    return model

def two_Network_CNN_inputIn(model_1, model_2, size = (256, 256, 5)):
    def pixel_sum(input):
        return input[:,:,:,0] + input[:,:,:,1]

    input_layer = Input(size)
    # model1 = Unet(size)
    # model2 = Unet(size)


    out1 = model_1(input_layer)
    out2 = model_2(input_layer)
    # con = concatenate([out1, out2])
    # flat = Flatten()(con)
    # scale = Dense(2, activation="softmax")(flat)
    # out = Add()([Multiply()([out1, scale[0]]), Multiply()([out1, scale[1]])])

    con = concatenate([out1, out2, input_layer])
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(con)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv_out = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv)

    model = Model(inputs=input_layer, outputs=conv_out)
    return model

def two_Network_CNNUpFeature(model_1, model_2, size = (256, 256, 5), feature_used = [1, 1, 1, 1, 1]):
    def up(x, times):
        for time in range(times):
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(int(x.shape[-1]), 1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    feature = []
    if feature_used[0]:
        feature5 = concatenate([model_1.get_layer("model1feature5-2").output, model_2.get_layer("model2feature5-2").output])
        feature5 = up(x=feature5, times=4)
        feature.append(feature5)

    if feature_used[1]:
        feature6 = concatenate(
            [model_1.get_layer("model1feature6-2").output, model_2.get_layer("model2feature6-2").output])
        feature6 = up(x=feature6, times=3)
        feature.append(feature6)

    if feature_used[2]:
        feature7 = concatenate(
            [model_1.get_layer("model1feature7-2").output, model_2.get_layer("model2feature7-2").output])
        feature7 = up(x=feature7, times=2)
        feature.append(feature7)

    if feature_used[3]:
        feature8 = concatenate(
            [model_1.get_layer("model1feature8-2").output, model_2.get_layer("model2feature8-2").output])
        feature8 = up(x=feature8, times=1)
        feature.append(feature8)

    if feature_used[4]:
        feature9 = concatenate(
            [model_1.get_layer("model1feature9-2").output, model_2.get_layer("model2feature9-2").output])
        feature.append(feature9)

    if sum(feature_used) > 1:
        con = concatenate(feature)
    else:
        con = feature[0]

    print(con.shape)

    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(con)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv_out = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv)

    model = Model(inputs=[model_1.input, model_2.input], outputs=conv_out)
    return model

def two_Network_CNNUpFeature_V2(model_1, model_2, size = (256, 256, 5), up_times = 0):
    def up(x, times):
        for time in range(times):
            x = UpSampling2D(size=(2, 2))(x)
            x = Conv2D(int(x.shape[-1]), 1, activation='relu', padding='same', kernel_initializer='he_normal')(x)
        return x

    input_layer = Input(size)

    out1 = model_1(input_layer)
    out2 = model_2(input_layer)

    if up_times:
        concate_up = up(concatenate([out1, out2]), times= up_times)
    else:
        concate_up = concatenate([out1, out2])

    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concate_up)
    conv = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv_out = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv)

    model = Model(inputs=input_layer, outputs=conv_out)
    return model


"""-------------------Self Model---------------------------------------"""
# self definition
def residual_UNet(input_size=(256, 256, 5)):
    def dense_block(x, filters, kernel, activation='relu'):
        conv1 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(x)
        conv2 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)

        output = keras.layers.Add()([x, conv2])
        return output

    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = dense_block(conv2, 128, 3)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = dense_block(conv3, 256, 3)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = dense_block(conv4, 512, 3)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv4 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = dense_block(conv4, 1024, 3)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = dense_block(conv6, 512, 3)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = dense_block(conv7, 256, 3)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = dense_block(conv8, 128, 3)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = dense_block(conv9, 64, 3)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

# self definition
def proposal_UNet_layer1_layerBlock(block, input_size=(256, 256, 5), classes=1, n_layers_per_block = 4, block_layers = 5):
    print("block layer num = {}".format(block_layers))

    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    if block_layers >= 1:
        layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    if block_layers >= 2:
        layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    if block_layers >= 3:
        layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    if block_layers >= 4:
        layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    if block_layers >= 5:
        layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    if block_layers >= 4:
        layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    if block_layers >= 3:
        layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    if block_layers >= 2:
        layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    if block_layers >= 1:
        layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    else:
        layer9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def UNet_DtoU5(block, name, input_size=(256, 256, 5), classes=1, n_layers_per_block = 8, block_num = 1):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-1")
    for i in range(block_num-1):
        layer1 = block(layer1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-" + str(i + 2))
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature2-1")
    for i in range(block_num-1):
        layer2 = block(layer2, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature2-" + str(i + 2))
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-1")
    for i in range(block_num-1):
        layer3 = block(layer3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-" + str(i + 2))
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature4-1")
    for i in range(block_num-1):
        layer4 = block(layer4, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature4-" + str(i + 2))
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-1")
    for i in range(block_num-1):
        layer5 = block(layer5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-" + str(i + 2))
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature6-1")
    for i in range(block_num-1):
        layer6 = block(layer6, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature6-" + str(i +2))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature7-1")
    for i in range(block_num-1):
        layer7 = block(layer7, 256, 3, n_layers_per_block=n_layers_per_block, name= name+"feature7-" + str(i +2))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature8-1")
    for i in range(block_num-1):
        layer8 = block(layer8, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature8-" + str(i +2))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature9-1")
    for i in range(block_num-1):
        layer9 = block(layer9, 64, 3, n_layers_per_block=n_layers_per_block, name= name+"feature9-" + str(i +2))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def UNet_DtoU5_2RDB_Residual2Block(block, name, input_size=(256, 256, 5), classes=1, n_layers_per_block = 8):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-1")
    layer1 = block(layer1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-2")
    layer1 = Add(name="feature1_Add")([conv1, layer1])
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature2-1")
    layer2 = block(layer2, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature2-2")
    layer2 = Add(name="feature2_Add")([conv2, layer2])
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-1")
    layer3 = block(layer3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-2")
    layer3 = Add(name="feature3_Add")([conv3, layer3])
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature4-1")
    layer4 = block(layer4, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature4-2")
    layer4 = Add(name="feature4_Add")([conv4, layer4])
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-1")
    layer5 = block(layer5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-2")
    layer5 = Add(name="feature5_Add")([conv5, layer5])
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature6-1")
    layer6 = block(layer6, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature6-2")
    layer6 = Add(name="feature6_Add")([conv6, layer6])

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature7-1")
    layer7 = block(layer7, 256, 3, n_layers_per_block=n_layers_per_block, name= name+"feature7-2")
    layer7 = Add(name="feature7_Add")([conv7, layer7])

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature8-1")
    layer8 = block(layer8, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature8-2")
    layer8 = Add(name="feature8_Add")([conv8, layer8])

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature9-1")
    layer9 = block(layer9, 64, 3, n_layers_per_block=n_layers_per_block, name= name+"feature9-2")
    layer9 = Add(name="feature9_Add")([conv9, layer9])
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def UNet_DtoU5_2RDB_ResidualMultiBlock(block, name, input_size=(256, 256, 5), classes=1, n_layers_per_block = 8):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-1")
    layer1_A = Add(name="feature1_1_Add")([conv1, layer1])
    layer1 = block(layer1_A, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-2")
    layer1_A = Add(name="feature1_2_Add")([layer1, layer1_A])
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1_A)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature2-1")
    layer2_A = Add(name="feature2_1_Add")([conv2, layer2])
    layer2 = block(layer2_A, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature2-2")
    layer2_A = Add(name="feature2_2_Add")([layer2_A, layer2])
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2_A)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-1")
    layer3_A = Add(name="feature3_1_Add")([conv3, layer3])
    layer3 = block(layer3_A, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-2")
    layer3_A = Add(name="feature3_2_Add")([layer3_A, layer3])
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3_A)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature4-1")
    layer4_A = Add(name="feature4_1_Add")([conv4, layer4])
    layer4 = block(layer4_A, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature4-2")
    layer4_A = Add(name="feature4_2_Add")([layer4_A, layer4])
    drop4 = Dropout(0.5)(layer4_A)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-1")
    layer5_A = Add(name="feature5_1_Add")([conv5, layer5])
    layer5 = block(layer5_A, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-2")
    layer5_A = Add(name="feature5_2_Add")([layer5_A, layer5])
    drop5 = Dropout(0.5)(layer5_A)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature6-1")
    layer6_A = Add(name="feature6_1_Add")([conv6, layer6])
    layer6 = block(layer6_A, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature6-2")
    layer6_A = Add(name="feature6_2_Add")([layer6_A, layer6])

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6_A))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature7-1")
    layer7_A = Add(name="feature7_1_Add")([conv7, layer7])
    layer7 = block(layer7_A, 256, 3, n_layers_per_block=n_layers_per_block, name= name+"feature7-2")
    layer7_A = Add(name="feature7_2_Add")([layer7_A, layer7])

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7_A))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature8-1")
    layer8_A = Add(name="feature8_1_Add")([conv8, layer8])
    layer8 = block(layer8_A, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature8-2")
    layer8_A = Add(name="feature8_2_Add")([layer8_A, layer8])

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8_A))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature9-1")
    layer9_A = Add(name="feature9_1_Add")([conv9, layer9])
    layer9 = block(layer9_A, 64, 3, n_layers_per_block=n_layers_per_block, name= name+"feature9-2")
    layer9_A = Add(name="feature9_2_Add")([layer9_A, layer9])
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9_A)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model
#
def UNet_DtoU5_feature(block, name, input_size=(256, 256, 5), classes=1, n_layers_per_block = 8, block_num = 1):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name + "feature1-1")
    for i in range(block_num-1):
        layer1 = block(layer1, 64, 3, n_layers_per_block=n_layers_per_block, name = name + "feature1-" + str(i + 2))
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name=name + "feature2-1")
    for i in range(block_num-1):
        layer2 = block(layer2, 128, 3, n_layers_per_block=n_layers_per_block, name = name + "feature2-" + str(i + 2))
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name + "feature3-1")
    for i in range(block_num-1):
        layer3 = block(layer3, 256, 3, n_layers_per_block=n_layers_per_block, name =name + "feature3-" + str(i + 2))
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name + "feature4-1")
    for i in range(block_num-1):
        layer4 = block(layer4, 512, 3, n_layers_per_block=n_layers_per_block, name = name + "feature4-" + str(i + 2))
    drop4 = Dropout(0.5)(layer4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name + "feature5-1")
    for i in range(block_num-1):
        layer5 = block(layer5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name + "feature5-" + str(i + 2))
    drop5 = Dropout(0.5)(layer5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name + "feature6-1")
    for i in range(block_num-1):
        layer6 = block(layer6, 512, 3, n_layers_per_block=n_layers_per_block, name= name + "feature6-" + str(i +2))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name + "feature7-1")
    for i in range(block_num-1):
        layer7 = block(layer7, 256, 3, n_layers_per_block=n_layers_per_block, name= name + "feature7-" + str(i +2))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name + "feature8-1")
    for i in range(block_num-1):
        layer8 = block(layer8, 128, 3, n_layers_per_block=n_layers_per_block, name= name + "feature8-" + str(i +2))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name + "feature9-1")
    for i in range(block_num-1):
        layer9 = block(layer9, 64, 3, n_layers_per_block=n_layers_per_block, name= name + "feature9-" + str(i +2))

    model = Model(input=input, output=layer9)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return (model, [layer5, layer6, layer7, layer8, layer9])

def UNet_2RDB(name, dropout= 0.5, input_size=(256, 256, 4), classes=1, n_layers_per_block = 8):

    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = RDBlocks(conv1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-1")
    layer1 = RDBlocks(layer1, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature1-2")
    pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = RDBlocks(conv2, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature2-1")
    layer2 = RDBlocks(layer2, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature2-2")
    pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = RDBlocks(conv3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-1")
    layer3 = RDBlocks(layer3, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature3-2")
    pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = RDBlocks(conv4, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature4-1")
    layer4 = RDBlocks(layer4, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature4-2")
    pool4 = MaxPooling2D((2, 2), None, 'same')(layer4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = RDBlocks(conv5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-1")
    layer5 = RDBlocks(layer5, 1024, 3, n_layers_per_block=n_layers_per_block, name = name+"feature5-2")


    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer5))
    merge6 = concatenate([layer4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = RDBlocks(conv6, 512, 3, n_layers_per_block=n_layers_per_block, name = name+"feature6-1")
    layer6 = RDBlocks(layer6, 512, 3, n_layers_per_block=n_layers_per_block, name= name+"feature6-2")

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer6))
    merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = RDBlocks(conv7, 256, 3, n_layers_per_block=n_layers_per_block, name = name+"feature7-1")
    layer7 = RDBlocks(layer7, 256, 3, n_layers_per_block=n_layers_per_block, name= name+"feature7-2")

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer7))
    merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = RDBlocks(conv8, 128, 3, n_layers_per_block=n_layers_per_block, name = name+"feature8-1")
    layer8 = RDBlocks(layer8, 128, 3, n_layers_per_block=n_layers_per_block, name= name+"feature8-2")

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(layer8))
    merge9 = concatenate([layer1, up9])
    conv9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = RDBlocks(conv9, 64, 3, n_layers_per_block=n_layers_per_block, name = name+"feature9-1")
    layer9 = RDBlocks(layer9, 64, 3, n_layers_per_block=n_layers_per_block, name= name+"feature9-2")
    drop9 = Dropout(dropout)(layer9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

# TU TD EXPERIMENT
def UNet_DtoU5_2RDB8(TU, TD, block, input_size=(256, 256, 4), classes=1, n_layers_per_block = 8, block_num = 2):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    layer1 = block(conv1, 64, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer1 = block(layer1, 64, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool1 = TransitionDown(layer1, 64)
    else:
        pool1 = MaxPooling2D((2, 2), None, 'same')(layer1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    layer2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer2 = block(layer2, 128, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool2 = TransitionDown(layer2, 128)
    else:
        pool2 = MaxPooling2D((2, 2), None, 'same')(layer2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    layer3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer3 = block(layer3, 256, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool3 = TransitionDown(layer3, 256)
    else:
        pool3 = MaxPooling2D((2, 2), None, 'same')(layer3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    layer4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer4 = block(layer4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(layer4)
    if TD:
        pool4 = TransitionDown(drop4, 512)
    else:
        pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    layer5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer5 = block(layer5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(layer5)

    if TU:
        merge6 = TransitionUp(drop4, drop5, 512)
    else:
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    layer6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer6 = block(layer6, 512, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge7 = TransitionUp(layer3, layer6, 256)
    else:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(layer6))
        merge7 = concatenate([layer3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    layer7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer7 = block(layer7, 256, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge8 = TransitionUp(layer2, layer7, 128)
    else:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(layer7))
        merge8 = concatenate([layer2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    layer8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer8 = block(layer8, 128, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge9 = TransitionUp(layer1, layer8, 64)
    else:
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(layer8))
        merge9 = concatenate([layer1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    layer9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    for i in range(block_num-1):
        layer9 = block(layer9, 64, 3, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(layer9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_layer1(block, input_size=(256, 256, 5), classes=1, n_layers_per_block = 4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_layer2(block, input_size=(256, 256, 5), classes=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)
    dense6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense6))
    merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)
    dense7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense7))
    merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)
    dense8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(dense8))
    merge9 = concatenate([conv1, up9])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = block(conv6, 64, 2, n_layers_per_block=n_layers_per_block)
    dense9 = block(conv9, 64, 2, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_TDTU_layer1(block, input_size=(256, 256, 5), classes=1, TD=1, TU=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    if TD:
        pool1 = TransitionDown(conv1, 64)
    else:
        pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)


    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool2 = TransitionDown(dense2, 128)
    else:
        pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool3 = TransitionDown(dense3, 256)
    else:
        pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    if TD:
        pool4 = TransitionDown(drop4, 512)
    else:
        pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    if TU:
        merge6 = TransitionUp(drop4, drop5, 512)
    else:
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    dense6 = block(conv6, 512, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge7 = TransitionUp(dense3, dense6, 256)
    else:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense6))
        merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    dense7 = block(conv7, 256, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge8 = TransitionUp(dense2, dense7, 128)
    else:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense7))
        merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    dense8 = block(conv8, 128, 3, n_layers_per_block=n_layers_per_block)

    if TU:
        merge9 = TransitionUp(conv1, dense8, 64)
    else:
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense8))
        merge9 = concatenate([conv1, up9])
    conv9 =  Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    dense9 = block(conv9, 64, 3, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

def proposal_UNet_TDTU_layer2(block, input_size=(256, 256, 5), classes=1, TD=1, TU=1, n_layers_per_block=4):
    input = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    if TD:
        pool1 = TransitionDown(conv1, 64)
    else:
        pool1 = MaxPooling2D((2, 2), None, 'same')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    dense2 = block(conv2, 128, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool2 = TransitionDown(dense2, 128)
    else:
        pool2 = MaxPooling2D((2, 2), None, 'same')(dense2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    dense3 = block(conv3, 256, 3, n_layers_per_block=n_layers_per_block)
    if TD:
        pool3 = TransitionDown(dense3, 256)
    else:
        pool3 = MaxPooling2D((2, 2), None, 'same')(dense3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    dense4 = block(conv4, 512, 3, n_layers_per_block=n_layers_per_block)
    drop4 = Dropout(0.5)(dense4)
    if TD:
        pool4 = TransitionDown(drop4, 512)
    else:
        pool4 = MaxPooling2D((2, 2), None, 'same')(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    dense5 = block(conv5, 1024, 3, n_layers_per_block=n_layers_per_block)
    drop5 = Dropout(0.5)(dense5)

    if TU:
        merge6 = TransitionUp(drop4, drop5, 512)
    else:
        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)
    dense6 = block(conv6, 512, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge7 = TransitionUp(dense3, dense6, 256)
    else:
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense6))
        merge7 = concatenate([dense3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)
    dense7 = block(conv7, 256, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge8 = TransitionUp(dense2, dense7, 128)
    else:
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense7))
        merge8 = concatenate([dense2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)
    dense8 = block(conv8, 128, 2, n_layers_per_block=n_layers_per_block)

    if TU:
        merge9 = TransitionUp(conv1, dense8, 64)
    else:
        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(dense8))
        merge9 = concatenate([conv1, up9])
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = block(conv6, 64, 2, n_layers_per_block=n_layers_per_block)
    dense9 = block(conv9, 64, 2, n_layers_per_block=n_layers_per_block)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(dense9)
    conv10 = Conv2D(classes, 1, activation='sigmoid')(conv9)

    model = Model(input=input, output=conv10)

    # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
    # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
    return model

"""-------------------Block---------------------------------------"""

def residual_block(x, filters, kernel, name, n_layers_per_block = 0, activation='relu'):
    conv1 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(x)
    conv2 = Conv2D(filters, kernel, activation=activation, padding='same', kernel_initializer='he_normal')(conv1)
    output = keras.layers.Add(name= name)([x, conv2])
    return output

def dense_block(x, filters, kernel, n_layers_per_block = 4, dropout_p = 0.2, growth_rate = 16):
    stack = x
    n_filters = filters

    for j in range(n_layers_per_block - 1):
        l = BN_ReLU_Conv(stack, growth_rate, filter_size= kernel, dropout_p=dropout_p)
        stack = concatenate([stack, l])
        n_filters += growth_rate

    l = BN_ReLU_Conv(stack, growth_rate, filter_size=kernel, dropout_p=dropout_p)
    stack = concatenate([stack, l])

    # skip_connection = stack
    # stack = TransitionDown(stack, filters, dropout_p)
    return stack

# https://github.com/rajatkb/RDNSR-Residual-Dense-Network-for-Super-Resolution-Keras/blob/master/main.py
def RDBlocks(x, filters, kernel, name, n_layers_per_block=4, g=16):
    li = [x]
    pas = Convolution2D(filters=g, kernel_size=(kernel, kernel), strides=(1, 1), padding='same', activation='relu')(x)

    for i in range(n_layers_per_block - 1):
        li.append(pas)
        out = Concatenate()(li)  # conctenated out put
        pas = Convolution2D(filters=g, kernel_size=(kernel, kernel), strides=(1, 1), padding='same', activation='relu')(out)

    li.append(pas)
    out = Concatenate()(li)
    feat = Convolution2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(out)

    feat = Add(name= name)([feat, x])
    return feat

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''

    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l

def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D((2,2))(l)
    return l

def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
    #Upsample and concatenate with skip connection
    l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(block_to_upsample)
    l = concatenate([l, skip_connection], axis=-1)
    return l

if __name__ == '__main__':
    x = np.load("E:\\allen\\data\\npy\\V3.2_x_7132in1783.npy")

    model = load_model("C:\\Users\\user\\Downloads\\20201207_256_7131_V3.2_HRNet_CE.h5")
    first = time.time()
    model.predict(x)
    second = time.time()
    print(second - first)

    model = load_model("C:\\Users\\user\\Downloads\\20201209_256_7131_V3.2_UNet_CE.h5")
    first = time.time()
    model.predict(x)
    second = time.time()
    print(second - first)

    model = segnet((256, 256, 4),
                    n_labels=1,
                    kernel=3,
                    pool_size=(2, 2),
                    output_mode="sigmoid")
    model.load_weights("C:\\Users\\user\\Downloads\\20210209_256(50%)_7131_V3.2_SegNet_CE-weights-epoch024-loss0.22139-val_loss0.34735.h5")
    first = time.time()
    model.predict(x)
    second = time.time()
    print(second - first)
    # model = DV3.Deeplabv3(weights=None, input_shape=(256, 256, 4), classes=1, backbone='mobilenetv2',
    #           OS=16, alpha=1., activation=None)
    # model.load_weights("E:\\allen\\data\\result\\paper_used_data\\other_model_experiment\\20200115_256(50%)_7060_V3.2_DV3_bin.h5")
    #

    # model_build = IBCNN.BCNN(base_estimator= model_select,
    #                          n_estimators= 10,
    #                          learning_rate= 2,
    #                          first_epochs= 5,
    #                          else_epochs= 5,
    #                          input_size= input_size,
    #                          output_size= (256, 256,1),
    #                          name= name[i],
    #                          batch_size= 3)
    # model_build = model.model(model=model_select, name=name[i], size=input_size)
    # model_build = GAN.Pix2Pix(generator= model_select, name= name[i])
    # model_build = model.AdaBoostCNN(base_estimator= model_select,
    #                                 n_estimators= 10,
    #                                 learning_rate= 2,
    #                                 epochs= 5,
    #                                 input_size= input_size,
    #                                 output_size= (256, 256, 1),
    #                                 batch_size= batch,
    #                                 name= name[i])

    # test
    # estimator_head = "20201029_256(50%)_18591_NRR(degree-90&180)_testNRR_UNet(2RDB8-DtoU-5)-originalReduceLR-sampleWeights_CE_"
    # estimator_path_head = "NRR_test\\"
    # estimator_path = [estimator_path_head + estimator_head + "0",
    #                   estimator_path_head + estimator_head + "1",
    #                   estimator_path_head + estimator_head + "2",
    #                   estimator_path_head + estimator_head + "3",
    #                   estimator_path_head + estimator_head + "4",
    #                   estimator_path_head + estimator_head + "5",
    #                   estimator_path_head + estimator_head + "6",
    #                   estimator_path_head + estimator_head + "7",
    #                   estimator_path_head + estimator_head + "8",
    #                   estimator_path_head + estimator_head + "9"]
    # model_build.test_IB(train_x, train_y, test_x, test_y,
    #                     data_start=test_data_start,
    #                     estimator_path=estimator_path,
    #                     save_path=name[i],
    #                     train_batch=3)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # model = Unet(size=(256, 256, 4))

# def dense_Unet(size= ( 256, 256, 4)):
#     def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
#         '''Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0)'''
#
#         l = BatchNormalization()(inputs)
#         l = Activation('relu')(l)
#         l = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(l)
#         if dropout_p != 0.0:
#             l = Dropout(dropout_p)(l)
#         return l
#
#     def TransitionDown(inputs, n_filters, dropout_p=0.2):
#         """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """
#         l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
#         l = MaxPooling2D((2, 2))(l)
#         return l
#
#     def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
#         '''Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection'''
#         # Upsample and concatenate with skip connection
#         l = Conv2DTranspose(n_filters_keep, kernel_size=3, strides=2, padding='same', kernel_initializer='he_uniform')(
#             block_to_upsample)
#         l = concatenate([l, skip_connection], axis=-1)
#         return l
#
#     def dense_block(inputs, )
#
#     input = Input(size)
#     conv1 = Conv2D(64, 3, activation= 'relu', padding= 'same')(input)
#     conv1 = BN_ReLU_Conv(conv1, 64)
#     TD1 = TransitionDown(conv1, 64)
#
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D((2,2), None, 'same')(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D((2,2), None, 'same')(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D((2,2), None, 'same')(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6])
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7])
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8])
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9])
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid', name="conv10")(conv9)
#
#     model = Model(input = input, output = conv10)
#
#     # # model.compile(optimizer = Adam(lr = 1e-4), loss = [loss.focal_plus_crossentropy_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
#     # # model.compile(optimizer = Adam(lr = 1e-4), loss = loss.focal_plus_cross, metrics = ['accuracy'])
#     # model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
#     return model


# def pspnet_decoder(f, n_classes)
#
# if __name__ == '__main__':
#     img_input, f = mobilenet_encoder()
#     img_output = segnet_decoder(f[4], 1, n_up=5)
#
#     m = Model(img_input, img_output)
#     m.summary()
#


# if i == 0:
#     model_select = model.Unet((256, 256, 4))
# elif i == 1:
#     model_select = model.Unet_elu((256, 256, 4))
# elif i == 2:
#     model_select = model.segnet((256, 256, 4), n_labels=1, output_mode="sigmoid")
# elif i == 3:
#     model_select = model.dense_UNet((256, 256, 4))
# elif i == 4:
#     model_select = DV3.Deeplabv3(weights=None, input_shape=(256, 256, 4), classes=1)
# elif i == 5:
#     model_base = model.Unet((256, 256, 4))
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))
# elif i == 6:
#     model_base = model.Unet_elu((256, 256, 4))
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))
# elif i == 7:
#     model_base = model.segnet((256, 256, 4), n_labels=1, output_mode="sigmoid")
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))
# elif i == 8:
#     model_base = model.dense_UNet((256, 256, 4))
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))
# else:
#     model_select = DV3.Deeplabv3(weights=None, input_shape=(256, 256, 4), classes=1)
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))
# else
#     model_select = FCDN.Tiramisu(input_shape=(256, 256, 4), n_classes=1, n_pool=5)
# else
#     model_base = FCDN.Tiramisu(input_shape=(256, 256, 4), n_classes=1, n_pool=5)
#     model_select = model.two_Network(model_base, model_base, size=(256, 256, 4))

# loading model
# model_select = load_model('.\\result\model_record\\' + name[0] + '_epoch_10' + '.h5',
#                           {'focal_plus_cross': loss.focal_plus_cross})

# model_select = load_model('.\\result\model_record\\' + name + '.h5',
#                           {'MaxPoolingWithArgmax2D': layers.MaxPoolingWithArgmax2D,
#                            'MaxUnpooling2D': layers.MaxUnpooling2D})

# model_select = model_DV3Plus.Deeplabv3(weights=None, input_shape=(256, 256, 5), classes=1)
# model_select.load_weights('.\\result\model_record\\' + name[i] + '.h5')

# compile
# model_select.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
# model_select.compile(optimizer=Adam(lr=1e-4), loss=[loss.focal_loss, loss.focal_loss, loss.focal_loss], metrics=['accuracy'])