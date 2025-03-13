# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:59:56 2020

@author: love_wx
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt 

CAPTCHA_LEN = 4

MODEL_SAVE_PATH = 'D:/ASIA/AI/lastwork/captcha/models/'
TEST_IMAGE_PATH = 'D:/ASIA/AI/lastwork/captcha/test/'

def get_image_data_and_name(fileName, filePath=TEST_IMAGE_PATH): #讀取測試的圖片
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #轉為灰諧
    img = img.convert("L")       
    image_array = np.array(img)    
    image_data = image_array.flatten()/255
    image_name = fileName[0:CAPTCHA_LEN]
    return image_data, image_name

def digitalStr2Array(digitalStr): #讀取圖片的實際值存進list
    digitalList = []
    for c in digitalStr:
        digitalList.append(ord(c) - ord('0')) #傳入字元，回傳對應的Unicode字元。
    return np.array(digitalList)

def model_test(): #測試模型
    nameList = []
    for pathName in os.listdir(TEST_IMAGE_PATH):
        nameList.append(pathName.split('/')[-1])
    totalNumber = len(nameList)
    #載入graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"crack_captcha.model-34400.meta")#載入模型
    graph = tf.get_default_graph() #獲取當前預設的計算圖
    #從graph取得 tensor，他們的name是在構建graph時定義的(檢視上一個的程式碼)
    input_holder = graph.get_tensor_by_name("data-input:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess: #在計算圖中讀取變數的值
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))#如果存在就從模型中恢復變數
        count = 0
        for fileName in nameList: #讀取圖片進行預測
            img_data, img_name = get_image_data_and_name(fileName, TEST_IMAGE_PATH)
            
            #輸出預測值
            predict = sess.run(predict_max_idx, feed_dict={input_holder:[img_data], keep_prob_holder : 1.0})
            filePathName = TEST_IMAGE_PATH + fileName
            print(filePathName)
            img = Image.open(filePathName)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            predictValue = np.squeeze(predict)#轉換向量組數，讓圖片正常顯示。
            rightValue = digitalStr2Array(img_name) #實際值
            if np.array_equal(predictValue, rightValue):
                result = '正確'
                count += 1
            else: 
                result = '錯誤'            
            print('實際值：{}， 預測值：{}，測試結果：{}'.format(rightValue, predictValue, result))
            print('\n')
        print('正確率：%.2f%%(%d/%d)' % (count*100/totalNumber, count, totalNumber))

if __name__ == '__main__':
    model_test()