import os
import sys
import urllib.request as request
from urllib.request import urlretrieve
import socket
import requests
import cv2

'''
下载Pubfig人脸数据集，下载下来之后根据坐标裁剪人脸，
并且每个人的人脸图片放到单独的一个文件夹中
'''

eval_urls = "FacialRecognition/dataset/url_people/eval_urls.txt"
dev_urls = "FacialRecognition/dataset/url_people/dev_urls.txt"

originDir = "./originDir2"  # 用来保存下载的原始图片
faceDir = "./faceDir"  # 用来保存裁剪的人脸子图

# 设定一下无响应时间，
timeout = 3
socket.setdefaulttimeout(timeout)

# 为请求增加Head
user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'
headers = ('User-Agent', user_agent)
opener = request.build_opener()
opener.addheaders = [headers]
request.install_opener(opener)

'''
    根据url获取文件名字
'''


def getPic(dir, name, nameNum, url, box):
    if not os.path.exists(dir):  # 每个人的图片放到一个单独的文件夹中，
        os.makedirs(dir, exist_ok=True)
    bad_url = []
    coord = box.split(",")
    print("coord:", coord)
    try:
        fileName = dir + "/" + name + "_" + nameNum + "_" + coord[0] + "_" + coord[1] + "_" + coord[2] + "_" + coord[
            3] + ".jpg"
        print("url::", url)
        print("fileName::", fileName)
        request.urlretrieve(url, fileName)
        # urlretrieve(url, fileName)
    except Exception as e:
        print(Exception, ':', e)
        bad_url.append(url)


def downloadPic(txtName):
    with open(txtName) as f:
        lineCount = 0
        nameCount = 0  # 用来给图片命名，每个图片的名字为：人名_nameCount
        lines = f.readlines()
        for line in lines:
            # print("line::", line)
            if lineCount >= 2:  # txt文件前面两行为数据格式说明，非有效数据，过滤掉。
                # name1, name2, number, url, box, md5 = line.split()  #有的人名有3个单词，这样会报错
                lineList = line.split()
                if 6 == len(lineList):
                    dir = lineList[0]
                    nameNum = lineList[2]
                    url = lineList[3]
                    box = lineList[4]
                if 7 == len(lineList):
                    dir = lineList[0]
                    nameNum = lineList[3]
                    url = lineList[4]
                    box = lineList[5]
                getPic(originDir + "/" + dir, dir, nameNum, url, box)  # 目录传进去的是originDir + "/" + dir,
            lineCount = lineCount + 1


if __name__ == '__main__':
    Barack = 'FacialRecognition/dataset/url_people/Barack.txt'
    downloadPic(Barack)
    # downloadPic(dev_urls)
    # downloadPic(eval_urls)



