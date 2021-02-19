'''
    canny ：21-21*5
    canny input image
        1. 灰階
        2. uint8
        3. threshold ratio
    canny output image
        1. max 255, min 0

    cv2.createTrackbar("滑軌名稱", "視窗名稱", min value, max value, 副函數名稱)

'''
import rasterio
import numpy as np
import os
import excel
import shutil
# import Augmentor
import cv2
# 確定 rasterio 讀取後的通道順序為 GBR

# # setup the path of data
# train_path = "D:\\allen\project\map\dataset\dataset_V2.0"
# test_path = "D:\\allen\project\map\dataset\dataset_V2.0"
#
# # parameter
# (channels, heights, widths) = (4, 256, 256)
# get the data
"""
從 dataset檔案讀取轉換成可以輸入模型的數據，要改變load_path位置
*****  變數
dataset：dataset檔案名稱
start_num：從第幾個資料(數字)開始
total_num：從start_num開始，總共讀取多少資料
size：衛星影像的大小
*---------------------------------
return (x, y)
"""
def load_data(dataset, start_num, total_num, size= (256, 256, 4)):
    # 要確認dataset檔案位置
    load_path = "E:\\allen\\data\\dataset\\"
    print('start load data')
    print('x')
    # variable
    train_x = np.zeros([total_num, size[0], size[1], size[2]], dtype=np.float32)
    train_y = np.zeros([total_num, size[0], size[1], 1], dtype=np.float32)

    # 讀取所有 train x 檔案
    for index in range(start_num, total_num + start_num):
        if ( index - start_num) % 100 == 0:
            print('x data：{}'.format(( index - start_num) / 100))
        # Read file
        data_raster = rasterio.open(load_path + dataset + '\\x\\' + str(index) + '.tif')
        # Read image
        data_raster_img = data_raster.read().transpose((1, 2, 0))
        # Normalize
        train_x[(index - start_num)] = (data_raster_img.astype('float32') / np.max(data_raster_img))

    print('y')
    # 讀取所有 train y 檔案
    for index in range(start_num, total_num+ start_num):
        if( index - start_num) % 100 == 0:
            print('y data：{}'.format(( index - start_num) / 100))
        # Read file
        data_raster = rasterio.open(load_path + dataset + '\\y\\' + str(index) + '.tif')
        # Read mask
        data_raster_nr = data_raster.read(1)
        # Normalize：有值=0,0=0

        train_y[ index - start_num] = np.expand_dims(np.ones(data_raster_nr.shape, dtype= np.float32) * (data_raster_nr > 0), axis= -1)
        # for x in range(size[0]):
        #     for y in range(size[1]):
        #         if data_raster_nr[x, y] == 0:
        #             train_y[ index - start_num, x, y, 0] = 0.0
        #         else:
        #             train_y[ index - start_num, x, y, 0] = 1.0

    print('finish load file.')
    return (train_x, train_y)

def load_data_edge(dataset, start_num, total_num, size= (256, 256, 5)):
    # parameter
    (train_x, train_y) = load_data(dataset, start_num=start_num, total_num=total_num, size=(size[0], size[1],size[2]-1))

    print("Read " + dataset + "gray image.")
    gray_img = np.zeros([total_num, size[0], size[1]], dtype=np.float32)
    for file in range(start_num, start_num + total_num):
        print("Read gray image：" + str(file) + '.png')
        gray = cv2.imread(load_path + dataset + '\\gray\\' + str(file) + '.png', cv2.IMREAD_GRAYSCALE)
        gray_img[file - start_num] = (gray / 255).astype('float32')

    gray_img = np.expand_dims(gray_img, axis=-1)
    train_x = np.concatenate((train_x, gray_img), axis=-1)
    # # parameter
    # train_x = np.zeros([total_num, size[0], size[1], size[2]], dtype=np.float32)
    # (train_x[:,:,:,:4], train_y) = load_data(dataset, start_num=start_num, total_num=total_num, size=(size[0], size[1],size[2]-1))
    #
    # print("Read " + dataset + "gray image.")
    # for file in range(start_num, start_num + total_num):
    #     print("Read gray image：" + str(file) + '.png')
    #     gray = cv2.imread(load_path + dataset + '\\gray\\' + str(file) + '.png', cv2.IMREAD_GRAYSCALE)
    #     train_x[file - start_num, :,:,4] = (gray / 255).astype('float32')

    return (train_x, train_y)

def load_data_landEdge(start_num, total_num, size=(256, 256, 1)):
    result = np.zeros([total_num, size[0], size[1], size[2]], np.float32)

    for index in range(total_num):
        print("Land edge: {}".format(index + start_num))
        data_raster = rasterio.open(".\\dataset_V3.2-edge\\edge\\" + str(index + start_num) + ".tif")
        data_raster_edge = data_raster.read(1)
        _, binary_edge = cv2.threshold(data_raster_edge, 0, 1, cv2.THRESH_BINARY_INV)
        result[index] = np.expand_dims(binary_edge, axis=-1)
    return result

def load_data_invert_y(dataset, start_num, total_num, size= (256, 256, 4), edge = False):
    if edge:
        (load_x, load_y) = load_data_edge(dataset, start_num, total_num, size=size)
    else:
        (load_x, load_y) = load_data(dataset, start_num, total_num, size=size)
    (re_x, re_y) = (load_x, np.zeros([total_num, size[0], size[1], 2], dtype=np.float32))

    re_y[:,:,:,0] = load_y.reshape(total_num, size[0], size[1])
    re_y[:,:,:,1] = (((np.invert(load_y.astype('uint8') * 255)) / 255).astype('float32')).reshape(total_num, size[0], size[1])

    return (re_x, re_y)

"""
將dataset的x資料轉成可以看得RGB影像，要改變load_path位置
*****  變數
dataset：dataset檔案名稱
*---------------------------------
無回傳值
"""
def image_restore(dataset):
    # 要確認dataset檔案位置
    load_path = "E:\\allen\\data\\dataset\\"
    for file in os.listdir(load_path + dataset + "\\x"):
        # Read file
        print("Read " + file)
        data_raster = rasterio.open(load_path + dataset + '\\x\\' + file)
        # Read image
        data_raster_img = (data_raster.read().transpose((1, 2, 0)) / np.max(data_raster.read())) * 255
        cv2.imwrite(load_path + dataset + "\\image\\" + file.rstrip(".tif") + '.png', data_raster_img[:,:,0:3].astype('uint8'))
"""
    將多個資料庫資料進行合併成為一個資料庫
    
    dataset_path_list：要合併的資料庫列表，主要是資料庫路徑
    dataset_name_list：每個對應資料庫自訂義名稱
    dataset_totalNum_list：每個資料庫總筆數
    concate_dataset_path：要合併到的目標資料庫路徑
"""
def dataset_resave(dataset_path_list, dataset_name_list, dataset_totalNum_list, concate_dataset_path):
    max_totalNum = np.max(dataset_totalNum_list)
    file_count = 0

    for index in range(max_totalNum):
        if index < dataset_totalNum_list[0]:
            si_src_file = dataset_path_list[0] + "\\x\\" + str(index + 1) + ".tif"
            label_src_file = dataset_path_list[0] + "\\y\\" + str(index + 1) + ".tif"
            si_dst_file = concate_dataset_path + "\\x\\" + str(file_count + 1) + ".tif"
            label_dst_file = concate_dataset_path + "\\y\\" + str(file_count + 1) + ".tif"
            label_dst_mark_file = concate_dataset_path + "\\y_original_label\\" + str(file_count + 1) + "_" + dataset_name_list[0] + ".tif"
            shutil.copyfile(si_src_file, si_dst_file)
            shutil.copyfile(label_src_file, label_dst_file)
            shutil.copyfile(label_src_file, label_dst_mark_file)
            file_count +=1

        if index < dataset_totalNum_list[1]:
            si_src_file = dataset_path_list[1] + "\\x\\" + str(index + 1) + ".tif"
            label_src_file = dataset_path_list[1] + "\\y\\" + str(index + 1) + ".tif"
            si_dst_file = concate_dataset_path + "\\x\\" + str(file_count + 1) + ".tif"
            label_dst_file = concate_dataset_path + "\\y\\" + str(file_count + 1) + ".tif"
            label_dst_mark_file = concate_dataset_path + "\\y_original_label\\" + str(file_count + 1) + "_" + dataset_name_list[1] + ".tif"
            shutil.copyfile(si_src_file, si_dst_file)
            shutil.copyfile(label_src_file, label_dst_file)
            shutil.copyfile(label_src_file, label_dst_mark_file)
            file_count += 1
        if index < dataset_totalNum_list[2] and len(dataset_name_list) >= 3:
            si_src_file = dataset_path_list[2] + "\\x\\" + str(index + 1) + ".tif"
            label_src_file = dataset_path_list[2] + "\\y\\" + str(index + 1) + ".tif"
            si_dst_file = concate_dataset_path + "\\x\\" + str(file_count + 1) + ".tif"
            label_dst_file = concate_dataset_path + "\\y\\" + str(file_count + 1) + ".tif"
            label_dst_mark_file = concate_dataset_path + "\\y_original_label\\" + str(file_count + 1) + "_" + dataset_name_list[2] + ".tif"
            shutil.copyfile(si_src_file, si_dst_file)
            shutil.copyfile(label_src_file, label_dst_file)
            shutil.copyfile(label_src_file, label_dst_mark_file)
            file_count += 1

        if index < dataset_totalNum_list[3] and len(dataset_name_list) >= 4:
            si_src_file = dataset_path_list[3] + "\\x\\" + str(index + 1) + ".tif"
            label_src_file = dataset_path_list[3] + "\\y\\" + str(index + 1) + ".tif"
            si_dst_file = concate_dataset_path + "\\x\\" + str(file_count + 1) + ".tif"
            label_dst_file = concate_dataset_path + "\\y\\" + str(file_count + 1) + ".tif"
            label_dst_mark_file = concate_dataset_path + "\\y_original_label\\" + str(file_count + 1) + "_" + dataset_name_list[3] + ".tif"
            shutil.copyfile(si_src_file, si_dst_file)
            shutil.copyfile(label_src_file, label_dst_file)
            shutil.copyfile(label_src_file, label_dst_mark_file)
            file_count += 1

def image_canny_restore(dataset):
    for file in os.listdir(load_path + dataset + "\\x"):
        # Read file
        print("Read " + file)
        data_raster = rasterio.open(load_path + dataset + '\\x\\' + file)
        # Read image
        data_raster_img = (data_raster.read().transpose((1, 2, 0)) / np.max(data_raster.read())) * 255
        data_raster_gray = cv2.cvtColor(data_raster_img.astype('float32'), cv2.COLOR_RGB2GRAY)
        data_raster_canny = cv2.Canny((data_raster_gray).astype('uint8'),
                                      21,
                                      21 * 5)
        cv2.imwrite(load_path + dataset + "\\gray\\" + file.rstrip(".tif") + '.png', data_raster_canny)

def move_data():
    # 移動檔案
    i = 1
    for filename in os.listdir(dataset_path + "\\x\\output"):
        print(filename)

        if i <= 2000:
            path = dataset_path + "\\x_a"
        else:
            path = dataset_path + "\\y_a"
        shutil.move(dataset_path + "\\x\\output" + '\\' + filename, path)
        i += 1

def rename_data():
    i = 1
    for filename in os.listdir(dataset_path + "\\x_a"):
        os.rename(dataset_path + "\\x_a\\" + filename, dataset_path + "\\x_a\\a_" + str(i) + ".png")
        i += 1
    i = 1
    for filename in os.listdir(dataset_path + "\\y_a"):
        os.rename(dataset_path + "\\y_a\\" + filename, dataset_path + "\\y_a\\a_" + str(i) + ".png")
        i += 1

def trans_to_npy(dataset, data_start, data_range, size=(256, 256, 4)):
    # (x, y)=load_data_edge(dataset,data_start, data_range, size=size)
    # np.save('.\\npy\\' + dataset + "_x_canny_" + str(data_start) + "in" + str(data_range) + '.npy', x)
    # np.save('.\\npy\\' + dataset + "_y_canny_" + str(data_start) + "in" + str(data_range) + '.npy', y)

    (x, y)=load_data(dataset, data_start, data_range, size=size)
    np.save('.\\npy\\' + dataset + "_x_" + str(data_start) + "in" + str(data_range) + '.npy', x)
    np.save('.\\npy\\' + dataset + "_y_" + str(data_start) + "in" + str(data_range) + '.npy', y)

def transHSV(x):
    total_num = len(x)
    # print("Total number={}".format(total_num))
    x = (x * 255).astype("uint8")
    x_HSV = np.zeros([total_num, 256, 256, 3], dtype=np.float32)

    for index in range(total_num):
        trans_HSV = cv2.cvtColor(x[index, :, :, 0:3], cv2.COLOR_BGR2HSV)
        x_HSV[index, :, :, 0] = (trans_HSV[:, :, 0] / 179).astype("float32")
        x_HSV[index, :, :, 1] = (trans_HSV[:, :, 1] / 255).astype("float32")
        x_HSV[index, :, :, 2] = (trans_HSV[:, :, 2] / 255).astype("float32")

    return x_HSV

def transYCbCr(x):
    total_num = len(x)
    # print("Total number={}".format(total_num))
    x = (x * 255).astype("uint8")
    x_YCbCr = np.zeros([total_num, 256, 256, 3], dtype=np.float32)

    for index in range(total_num):
        trans_YCbCr = cv2.cvtColor(x[index, :, :, 0:3], cv2.COLOR_BGR2YCrCb)
        x_YCbCr[index] = (trans_YCbCr / 255).astype("float32")

    return x_YCbCr

def data_tag(y):
    one = np.ones([256, 256, 1], dtype= np.float32)

    for index in range(len(y)):
        if np.sum(y[index]):
            y[index] = one

    return y

# 移除掉沒有標記的資料
def remove_non_goal_img(x, y):
    new_y = []
    new_x = []

    # bri = np.where(np.sum(y, axis=(1, 2, 3)) != 0)
    # np.delete(new_y, bri)
    # np.delete(new_x, bri)


    for index in range(len(x)):
        if np.sum(y[index]) != 0:
            print(np.sum(y[index]))
            new_x.append(x[index])
            new_y.append(y[index])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    print("new x shape {}".format(new_x.shape))
    print("new y shape {}".format(new_y.shape))
    return (new_x, new_y)

# 根據結果，移除小於門檻的資料
def extract_low_result(x, y, excel_file, total_num, extract_index= "iou", threshold= 0.5):
    file = excel.Excel(file_path= excel_file)
    print("Excel file {} is opened.".format(excel_file))

    if extract_index == "iou" or extract_index == "IoU":
        index = file.read_excel(start= "c3:c" + str(2+total_num))
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

    index = np.array(index)

    print("Index is {}.".format(extract_index))
    high_result = np.array(np.where(index >threshold))

    extract_x = np.delete(x, high_result, axis= 0)
    extract_y = np.delete(y, high_result, axis= 0)

    return (extract_x, extract_y)

# 要給分類任務(2類，大於門檻值的資料及小於門檻值的資料)轉換 GT
def transfer_original_to_classification_task(excel_file, total_num, extract_index= "iou", threshold= 0.5):
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

    index = np.array(index, dtype= np.float32)

    print("Index is {}.".format(extract_index))
    print("Index shape {}".format(index.shape))
    print("Index dtype {}".format(index.dtype))
    low_result = np.ones(index.shape) * (index <= threshold)
    high_result = np.ones(index.shape) * (index > threshold)

    output_y = np.zeros((len(index), 2))
    output_y[:, 0] = high_result
    output_y[:, 1] = low_result

    return output_y

if __name__ == "__main__":
    dataset= "V1.0"
    start_num= 1
    total_num= 239777
    y = load_data(dataset, start_num, total_num, size=(256, 256, 4))

    check_y = np.sum(y, axis=0)
    yes_y = (check_y > 0) * np.ones(check_y.shape)
    no_y = (check_y == 0) * np.ones(check_y.shape)

    print(check_y.shape)
    print(yes_y.shape)
    print(no_y.shape)
    print(np.sum(yes_y))
    print(np.sum(no_y))

    # 儲存npy
    # trans_to_npy("R_256_r180", 1, 4058, size=(256, 256, 4))
    # trans_to_npy("R_256_r180", 4059, 1014, size=(256, 256, 4))

# 低準確資料提取
#                 low_test_x = np.delete(test_x, np.array(np.where(norm > threshold[c])), axis=0)
#                 low_test_y = np.delete(test_y, np.array(np.where(norm > threshold[c])), axis=0)
#                 # 高準確資料提取
#                 high_test_x = np.delete(test_x, np.array(np.where(norm <= threshold[c])), axis=0)
#                 high_test_y = np.delete(test_y, np.array(np.where(norm <= threshold[c])), axis=0)

