# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:45:03 2020

@author: love_wx
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import time


#驗證碼圖片的存放路徑
CAPTCHA_IMAGE_PATH = 'D:/ASIA/AI/lastwork/captcha/train/'
#驗證碼圖片的寬度
CAPTCHA_IMAGE_WIDHT = 160
#驗證碼圖片的高度
CAPTCHA_IMAGE_HEIGHT = 60

#[0,1,2,3,4,5,6,7,8,9]
CHAR_SET_LEN = 10
CAPTCHA_LEN = 4

#90%的驗證碼圖片放入訓練集中
TRAIN_IMAGE_PERCENT = 0.9
#訓練集，用於訓練的驗證碼圖片的檔名
TRAINING_IMAGE_NAME = []
#驗證集，用於模型驗證的驗證碼圖片的檔名
VALIDATION_IMAGE_NAME = []

MODEL_SAVE_PATH = 'D:/ASIA/AI/lastwork/captcha/models/'

def get_image_file_name(imgPath=CAPTCHA_IMAGE_PATH):
    fileName = []
    total = 0
    for filePath in os.listdir(imgPath):
        captcha_name = filePath.split('/')[-1]
        fileName.append(captcha_name)
        total += 1
    return fileName, total
#將驗證碼轉換為訓練時用的標籤向量，維數是 40   
#例如，如果驗證碼是 ‘0296’ ，則對應的標籤是
# [1 0 0 0 0 0 0 0 0 0
#  0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1
#  0 0 0 0 0 0 1 0 0 0]
def name2label(name):
    label = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i*CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label
    
#取得驗證碼圖片的資料以及它的標籤        
def get_data_and_label(fileName, filePath=CAPTCHA_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    #轉為灰度圖
    img = img.convert("L")       
    image_array = np.array(img)    
    image_data = image_array.flatten()/255
    image_label = name2label(fileName[0:CAPTCHA_LEN])
    return image_data, image_label
    
#生成一個batch    
def get_next_batch(batchSize=32, trainOrTest='train', step=0):
    batch_data = np.zeros([batchSize, CAPTCHA_IMAGE_WIDHT*CAPTCHA_IMAGE_HEIGHT])#[32,160*60]
    batch_label = np.zeros([batchSize, CAPTCHA_LEN * CHAR_SET_LEN])#[32,4*4]
    fileNameList = TRAINING_IMAGE_NAME
    if trainOrTest == 'validate':        
        fileNameList = VALIDATION_IMAGE_NAME
        
    totalNumber = len(fileNameList) 
    indexStart = step*batchSize    
    for i in range(batchSize):
        index = (i + indexStart) % totalNumber
        name = fileNameList[index]        
        img_data, img_label = get_data_and_label(name)
        batch_data[i, : ] = img_data
        batch_label[i, : ] = img_label  
    return batch_data, batch_label
    
#構建卷積神經網路並訓練
def train_data_with_CNN():
    #初始化權值
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var
    #初始化偏置    
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var
    #卷積    
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    #池化過濾器的大小為2*2, 移動步長為2，使用全0填充  
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)     

    #輸入層
    #data-input在下個程式測試model時會用到它
    #tf.placeholder 用于定義過程，在執行的时候再赋與具體的值
    #tf.placeholder(dtype, shape=None, name=None) shape,默认是None是一维值
    X = tf.placeholder(tf.float32, [None, CAPTCHA_IMAGE_WIDHT * CAPTCHA_IMAGE_HEIGHT], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN], name='label-input')    
    x_input = tf.reshape(X, [-1, CAPTCHA_IMAGE_HEIGHT, CAPTCHA_IMAGE_WIDHT, 1], name='x-input')
    #輸出結果之前使用 Dropout 函數避免過度配適
    #keep_prob在下個程式測試model時會用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    #第一層卷積(32 個神經元)，會利用解析度 5x5 的 filter 取出 32 個特徵，然後池化
    W_conv1 = weight_variable([5,5,1,32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #第二層卷積（64 個神經元)，會用解析度 5x5 的 filter 取出 32 個特徵，然後池化
    W_conv2 = weight_variable([5,5,32,64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2,'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #第三層卷積（64 個神經元)，會用解析度 5x5 的 filter 取出 64 個特徵，然後池化
    W_conv3 = weight_variable([5,5,64,64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    #全連結層（1024 個神經元)，會將圖片的 1024 個特徵攤平
    #每次池化後，圖片的寬度和高度均縮小為原來的一半，進過上面的三次池化，寬度和高度均縮小8倍
    W_fc1 = weight_variable([20*8*64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 20*8*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #輸出層(40個神經元）
    W_fc2 = weight_variable([1024, CAPTCHA_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CAPTCHA_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    #tf.reshape 矩陣變形
    predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN], name='labels')
    #預測結果
    # predict_max_idx在下個程式測試model時會用到它
    #tf.argmax 返回最大的那個數值的索引值
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    
    saver = tf.train.Saver() #建立物件來儲存變量
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) #初始化模型的参数
        steps = 0 #從0開始總共跑50000個，一次跑100個
        for epoch in range(50000):
            train_data, train_label = get_next_batch(100, 'train', steps)
            sess.run(optimizer, feed_dict={X : train_data, Y : train_label, keep_prob:0.85})
            #keep_prob保留元素
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(100, 'validate', steps)
                acc = sess.run(accuracy, feed_dict={X : test_data, Y : test_label, keep_prob:1.0})
                #feed_dict 给使用placeholder创建出来的tensor赋值
                print("steps=%d, accuracy=%f" % (steps, acc))
                if acc > 0.99:
                    saver.save(sess, MODEL_SAVE_PATH+"crack_captcha.model", global_step=steps)
                    #儲存模型
                    break
            steps += 1
if __name__ == '__main__':    
    image_filename_list, total = get_image_file_name(CAPTCHA_IMAGE_PATH)
    random.seed(time.time())
    #打亂順序
    random.shuffle(image_filename_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    #分成測試集
    TRAINING_IMAGE_NAME = image_filename_list[ : trainImageNumber]
    #和驗證集
    VALIDATION_IMAGE_NAME = image_filename_list[trainImageNumber : ]
    train_data_with_CNN()    
    print('Training finished')