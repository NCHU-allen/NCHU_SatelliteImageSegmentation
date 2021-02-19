import cv2
import estimate
import numpy as np
import excel
import os
import model
import rasterio
from keras.models import load_model, Model

def check_threshold(y_pred, size= (256, 256, 1), threshold= 0.5):
    y_out = np.zeros(((len(y_pred), size[0], size[1], size[2])), dtype= np.uint8)
    print("Check the threshold.")
    for index in range(len(y_pred)):
        for x in range(size[0]):
            for y in range(size[1]):
                if y_pred[index, x, y] > threshold:
                    y_out[index, x, y] = 1
                else:
                    y_out[index, x, y] = 0
        print("threshold image:{}".format(index+1))
    return y_out

def recover_area(img_predict, start, number, file_path = ".\\dataset_V3.2-edge\\edge\\", size = (256, 256, 1), threshold= 0.5):
    result = np.zeros([number, size[0], size[1], size[2]], np.uint8)
    kernel = np.ones((2, 2), np.uint8)

    for index in range(number):
        data_raster = rasterio.open(file_path + str(index + start) + ".tif")
        data_raster_edge = data_raster.read(1)
        _, binary_edge = cv2.threshold(data_raster_edge, 0, 255, cv2.THRESH_BINARY_INV)
        binary_edge = cv2.erode(binary_edge.astype("uint8"), kernel, iterations=1)

        _, contours, hierarchy = cv2.findContours(binary_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in range(len(contours)):
            land_area = np.zeros([size[0], size[1], size[2]], np.uint8)
            cv2.drawContours(land_area, contours, contour, 1, -1)
            union_area = np.sum(np.bitwise_and(land_area, img_predict[index]))

            if union_area / np.sum(land_area) <= threshold:
                print("recover low : {}".format(union_area / np.sum(land_area)))
                continue

            cv2.drawContours(result[index], contours, contour, 1, -1)

        result[index] = np.expand_dims(cv2.dilate(result[index], kernel, iterations=1), axis=-1)
    return result

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    (train_x, train_y) = (np.load(".\\npy\\V3.2_x_1in7131.npy"),
                          np.load(".\\npy\\V3.2_y_1in7131.npy"))

    model_select = model.UNet_DtoU5(block=model.RDBlocks,
                                    input_size=(256, 256, 4),
                                    n_layers_per_block=8,
                                    block_num= 2)
    model_select.load_weights(
        ".\\result\model_record\V3.2_test\\20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE\\20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE.h5")
    test_flag = 1
    batch = 3
    epoch = 50
    print("model building.")
    # model_build = model.model(model=model_select, name="20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE", size=(256, 256, 4))

    print("model building.")
    model_build = model.model(model=model_select, name="20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE", size=(256, 256, 4))

    model_build.test(x_test=train_x, y_test=train_y, data_start=1, batch_size=batch, save_path="20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE_trainData")


    # y_predict = model_select.predict(train_x, batch_size=batch)
    #
    # print("Check the threshold.")
    # y_output = check_threshold(y_predict,
    #                            size=(256, 256, 1),
    #                            threshold=0.5)
    # iou = estimate.IOU(train_y, y_output, 256, len(train_x))

    # ex = excel.Excel(".\\result\\data\\V3.2_test\\20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE_iou.xlsx")
    # iou = ex.read_excel(start="c3:c1785")
    # ex.close_excel()


    # for index in range(len(iou)):
    #     if iou[index] < 0.5:
    #         continue

        # print("Remove {}".format(7132 + index))
        # os.remove("F:\\allen\data\dataset\dataset_V3.2\\UNet(2RDB8-DtoU-5)_CE_less_05\\20200525_256(50%)_7131_V3.2_UNet(2RDB8-DtoU-5)_CE\\" + str(7132 + index) + ".png")

        # os.remove("F:\\allen\data\dataset\dataset_V3.2\\UNet(2RDB8-DtoU-5)_CE_less_05\\x\\" + str(1 + index) + ".tif")
        # os.remove(
        #     "F:\\allen\data\dataset\dataset_V3.2\\UNet(2RDB8-DtoU-5)_CE_less_05\\y\\" + str(1 + index) + ".tif")
        # os.remove(
        #     "F:\\allen\data\dataset\dataset_V3.2\\UNet(2RDB8-DtoU-5)_CE_less_05\\image\\" + str(1 + index) + ".png")

