import numpy as np
from keras.models import load_model, Model
import json
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
import estimate
import excel
import postprocessing
import cv2
import os
from keras.layers import *

class BCNN(object):
    def __init__(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError(
                '''AdaBoostClassifier can only be called with keyword
                   arguments for the following keywords: base_estimator ,n_estimators,
                    learning_rate,algorithm,random_state''')
        allowed_keys = ['base_estimator',
                        'n_estimators',
                        'learning_rate',
                        'first_epochs',
                        'else_epochs',
                        'input_size',
                        'output_size',
                        'name',
                        'batch_size']
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
            if 'first_epochs' in kwargs: first_epochs = kwargs.pop('first_epochs')
            if 'else_epochs' in kwargs: else_epochs = kwargs.pop('else_epochs')
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

        self.first_epochs = first_epochs
        self.else_epochs = else_epochs
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

        # dict_estimators_weights = {}
        # dict_estimators_weights['estimator_weights'] = self.estimator_weights_
        # with open(".\\result\\" + self.name + 'estimator_weights.json', 'w', encoding='utf-8') as f:
        #     json.dump(dict_estimators_weights, f)

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
        callBack = [saveModel, reduce_lr]

        if order == 0:
            estimator.fit(x, y, sample_weight= sample_weight, epochs= self.first_epochs, batch_size= self.batch_size,
                          validation_split=0.125, callbacks=callBack)
        else:
            estimator.fit(x, y, sample_weight=sample_weight, epochs=self.else_epochs, batch_size=self.batch_size,
                          validation_split=0.125, callbacks=callBack)
        estimator.save(".\\result\model_record\\" + self.name + '_' + str(order) + '.h5')

#         CNN 訓練資料預測
        y_pred = estimator.predict(x)
#         CNN 檢查預測結果比較
        y_threshold = y_pred >= threshold
        y_threshold = np.ones(y_threshold.shape, dtype=np.uint8) * y_threshold
        # if np.sum(y_threshold, axis=(0, 1, 2, 3)) == 0:
        #     return None, 0, None
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

    def test(self, x_test, y_test, data_start, estimator_path, save_path, batch_size= 10, threshold= 0.5):
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

        base_estimator = self.base_estimator
        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)

        base_estimator.load_weights(".\\result\\model_record\\" + estimator_path + ".h5")

        y_predict = base_estimator.predict(x_test, batch_size=batch_size)

        print("Check the threshold.\ny_test.shape = {}".format(y_test.shape))
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
        ex_iou.write_excel("e1", "precision", vertical=True)
        ex_iou.write_excel("e2", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        ex_iou.write_excel("g1", "recall", vertical=True)
        ex_iou.write_excel("g2", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        ex_iou.write_excel("i1", "F1", vertical=True)
        ex_iou.write_excel("i2", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def test_avr(self, x_test, y_test, data_start, estimator_path, save_path, batch_size= 10, threshold= 0.5):
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

        base_estimator = self.base_estimator

        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)

        print("Start test_avr")
        y_predict = np.zeros(y_test.shape, dtype= np.float32)
        for model_file in estimator_path:
            print("Read model weight {}".format(".\\result\\model_record\\" + model_file + ".h5"))
            base_estimator.load_weights(".\\result\\model_record\\" + model_file + ".h5")
            result = base_estimator.predict(x_test, batch_size=batch_size) / self.n_estimators
            y_predict = y_predict + result

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size=self.output_size,
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
        ex_iou.write_excel("e1", "precision", vertical=True)
        ex_iou.write_excel("e2", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        ex_iou.write_excel("g1", "recall", vertical=True)
        ex_iou.write_excel("g2", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        ex_iou.write_excel("i1", "F1", vertical=True)
        ex_iou.write_excel("i2", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def test_weights(self, x_train, y_train, x_test, y_test, data_start, estimator_path, save_path, threshold= 0.5, estimator_weights = None):
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

        base_estimator = self.base_estimator

        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)

        if estimator_weights == None:
            estimator_weights = self.update_weight(x_train, y_train, estimator_path, threshold=0.5)
            dict_estimators_weights = {}
            dict_estimators_weights['estimator_weights'] = estimator_weights
            # with open(".\\result\\" + self.name + '_estimator_weights.json', 'w', encoding='utf-8') as f:
            #     json.dump(dict_estimators_weights, f)

        y_predict = np.zeros(y_test.shape, dtype=np.float32)
        count = 0
        for model_file in estimator_path:
            print("Read model weight {}".format(".\\result\\model_record\\" + model_file + ".h5"))
            base_estimator.load_weights(".\\result\\model_record\\" + model_file + ".h5")
            result = base_estimator.predict(x_test, batch_size=self.batch_size) * estimator_weights[count]
            y_predict = y_predict + result
            count = count + 1

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size=self.output_size,
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
        ex_iou.write_excel("e1", "precision", vertical=True)
        ex_iou.write_excel("e2", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        ex_iou.write_excel("g1", "recall", vertical=True)
        ex_iou.write_excel("g2", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        ex_iou.write_excel("i1", "F1", vertical=True)
        ex_iou.write_excel("i2", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_AdaBWeightiou.xlsx")
        ex_iou.close_excel()

    def test_IB(self, x_train, y_train, x_test, y_test, data_start, estimator_path, save_path, train_batch, threshold= 0.5):
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

        base_estimator = self.base_estimator


        # if x_test.ndim == 3:
        #     x_test = np.expand_dims(x_test, axis=-1)
        # if y_test.ndim == 3:
        #     y_test = np.expand_dims(y_test, axis=-1)

        y_predict = np.zeros(y_test.shape, dtype= np.float32)
        estimator_weights = np.zeros(len(estimator_path))
        # estimator_weights = self.update_weight(x_train, y_train, estimator_path, threshold=0.5)

        print("Start IB training processing.")
        for index in range(len(x_train) // train_batch):
            print("Data for training {}".format(index +1))
            if index > len(x_train) // train_batch:
                x = x_train[index * train_batch:]
                y = y_train[index * train_batch:]
            else:
                x = x_train[index * train_batch:(index + 1) * train_batch]
                y = y_train[index * train_batch:(index + 1) * train_batch]
            print("x {}".format(x.shape))

            estomator_weights_update = self.update_weight(x, y, estimator_path, threshold= 0.5)
            if index == 0:
                estimator_weights = estomator_weights_update
            else:
                estimator_weights = (estimator_weights + index * estomator_weights_update) / (index + 1)
            estimator_weights = estimator_weights / np.sum(estimator_weights)

        print("Start predict.")
        count = 0
        for model_file in estimator_path:
            print("Read model weight {}".format(".\\result\\model_record\\" + model_file + ".h5"))
            base_estimator.load_weights(".\\result\\model_record\\" + model_file + ".h5")
            result = base_estimator.predict(x_test, batch_size=self.batch_size) * estimator_weights[count]
            y_predict = y_predict + result
            count = count + 1

        print("Check the threshold.")
        y_output = postprocessing.check_threshold(y_predict,
                                                  size=self.output_size,
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
        ex_iou.write_excel("e1", "precision", vertical=True)
        ex_iou.write_excel("e2", precision, vertical=True)
        ex_iou.write_excel("f1", "avr_precision", vertical=True)
        ex_iou.write_excel("f2", avr_precision, vertical=True)
        ex_iou.write_excel("g1", "recall", vertical=True)
        ex_iou.write_excel("g2", recall, vertical=True)
        ex_iou.write_excel("h1", "avr_recall", vertical=True)
        ex_iou.write_excel("h2", avr_recall, vertical=True)
        ex_iou.write_excel("i1", "F1", vertical=True)
        ex_iou.write_excel("i2", F1, vertical=True)
        ex_iou.write_excel("j1", "avr_F1", vertical=True)
        ex_iou.write_excel("j2", avr_F1, vertical=True)

        ex_iou.save_excel(file_name=".\\result\data\\" + save_path + "_iou.xlsx")
        ex_iou.close_excel()

    def update_weight(self, x, y, estimator_path, threshold = 0.5):
        base_model = self.base_estimator
        result_iou = np.zeros(len(estimator_path))
        num = 0
        avr_iou_threshold = np.sum(self.IOU(y, np.zeros(y.shape), number=len(y))) / len(y)

        print("Start update_weight function.")
        for estimator in estimator_path:
            # print("Loading Weights：{}".format(".\\result\\model_record\\" + estimator))
            base_model.load_weights(".\\result\\model_record\\" + estimator + ".h5")
            y_pred = base_model.predict(x)

            y_threshold = y_pred >= threshold
            y_result = np.ones(y_threshold.shape, dtype=np.uint8) * y_threshold

            avr_iou = np.sum(self.IOU(y, y_result, number= len(y))) / len(y)
            if avr_iou > avr_iou_threshold:
                result_iou[num] = avr_iou
            else:
                result_iou[num] = 0
            num = num + 1

        weights = result_iou / np.sum(result_iou)
        return weights

    def IOU(self, gt, predict, number):
        print("Start estimate IoU.")
        IOU = np.zeros(number)

        intersection = np.multiply(gt, predict)
        union = gt + predict
        union[union >= 1] = 1

        for index in range(number):
            intersection_sum = np.sum(intersection[index])
            union_sum = np.sum(union[index])

            if union_sum == 0:
                IOU[index] = 1
                continue
            else:
                IOU[index] = intersection_sum / union_sum

        return IOU