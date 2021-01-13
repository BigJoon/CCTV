#src 에 있는 파일들을 읽어와서 dst로 복사해주는 텔레포트 스크립트!(이 땐 jpg랑 xml 파일들 복사 해줄 때 썼음.)
# 아랫쪽에 있는 dataset_path 같은 경로는 따로 안썼음. 신경 안써도 됨.
# Made by YJHwang


import os
from xml.etree.ElementTree import parse
import shutil

##First Read "trainval.txt"
f = open('trainval.txt',mode='rt',encoding='utf-8')

line_num=1
line = f.readline()

while line:

    src = '/hdd/TMP_D/Resized_Data_Voc_For_Song/VOCdevkit/VOC2007/Annotations/'+line.rstrip('\n')+'.xml'
    #print(src)
    dst = '/hdd/Fresh_Data/Class8_A/VOCdevkit/VOC2007/Annotations/'
    shutil.copy(src,dst)
    line = f.readline()
    line_num +=1

f.close()
