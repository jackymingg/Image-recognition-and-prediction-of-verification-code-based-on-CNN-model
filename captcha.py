# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:21:16 2020

@author: love_wx
"""
import sys
import os
import shutil
import random
import time
from captcha.image import ImageCaptcha
#captcha是用於生成驗證碼圖片的庫，可以 pip install captcha 來安裝它

#用於生成驗證碼的字集
CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']
#字集的長度
CHAR_SET_LEN = 10
#驗證碼的長度，每個驗證碼由4個數字組成
CAPTCHA_LEN = 6

#驗證碼圖片的存放路徑
CAPTCHA_IMAGE_PATH = 'D:/ASIA/AI/lastwork/captcha/train/'
#用於模型測試的驗證碼圖片的存放路徑，它裡面的驗證碼圖片作為測試集
TEST_IMAGE_PATH = 'D:/ASIA/AI/lastwork/captcha/test/'
#用於模型測試的驗證碼圖片的個數，從生成的驗證碼圖片中取出來放入測試集中
TEST_IMAGE_NUMBER = 100

#生成驗證碼圖片，6位的十進位制數字可以有10000種驗證碼
def generate_captcha_image(charSet = CHAR_SET, charSetLen=CHAR_SET_LEN, captchaImgPath=CAPTCHA_IMAGE_PATH):   
    k  = 0
    total = 1
    for i in range(CAPTCHA_LEN):
        total *= charSetLen
    
    for i in range(charSetLen):
        for j in range(charSetLen):
            for m in range(charSetLen):
                for n in range(charSetLen):
                    captcha_text = charSet[i] + charSet[j] + charSet[m] + charSet[n]
                    image = ImageCaptcha()
                    image.write(captcha_text, captchaImgPath + captcha_text + '.jpg')
                    k += 1
                    sys.stdout.write("\rCreating %d/%d" % (k, total))
                    sys.stdout.flush()
                    
#從驗證碼的圖片集中取出一部分作為測試集，這些圖片不參加訓練，只用於模型的測試                    
def prepare_test_set():
    fileNameList = []    
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filePath.split('/')[-1]
        fileNameList.append(captcha_name)
    random.seed(time.time())
    random.shuffle(fileNameList) 
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.move(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)
                        
if __name__ == '__main__':
    generate_captcha_image(CHAR_SET, CHAR_SET_LEN, CAPTCHA_IMAGE_PATH)
    prepare_test_set()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()  