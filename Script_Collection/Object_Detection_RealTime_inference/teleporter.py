#src 에 있는 파일들을 읽어와서 dst로 복사해주는 텔레포트 스크립트!(이 땐 jpg랑 xml 파일들 복사 해줄 때 썼음.)
# 아랫쪽에 있는 dataset_path 같은 경로는 따로 안썼음. 신경 안써도 됨.
# Made by YJHwang


import os
from xml.etree.ElementTree import parse
import shutil
#여기엔 원본 데이터에서 VOC2007 까지의 경로를 넣어주면 될듯?
dataset_path = '/hdd/TMP_D/Resized_Data_Voc_For_Song/VOCdevkit/VOC2007'
image_dir_path = os.path.join(dataset_path,'JPEGImages')
annotations_dir_path = os.path.join(dataset_path, 'Annotations')
#trainval_txt_path = os.path.join(dataset_path, 'ImageSets/Main/trainval.txt')
#test_txt_path = os.path.join(dataset_path, 'ImageSets/Main/test.txt')

#TODO : trainval.txt 읽어와서 해당하는 파일들을 각각 Annotations 과 JPEGImages 에서 빼서
# /hdd/Fresh_Data/Class6/VOCdevkit/VOC2007/JPEGImages
# /hdd/Fresh_Data/Class6/VOCdevkit/VOC2007/Annotations
# 위 경로로 "복사"해야함.

##First Read "trainval.txt"
f = open('trainval.txt',mode='rt',encoding='utf-8')

line_num=1
line = f.readline()

while line:
    #print('%d %s' %(line_num,line))
    #print(line)  ==> line에 파일들 잘 들어가 잇음.
    #TODO 여기서 line 에 들어가 있는 파일이름으로 복사하는거 하면 됨. Annotation 이랑 JPEGImages로!!!
    
    #line.rstrip('\n')
    #print(line.rstrip('\n'))
    src = '/hdd/TMP_D/Resized_Data_Voc_For_Song/VOCdevkit/VOC2007/Annotations/'+line.rstrip('\n')+'.xml'
    #print(src)
    dst = '/hdd/Fresh_Data/Class8_A/VOCdevkit/VOC2007/Annotations/'
    shutil.copy(src,dst)
    line = f.readline()
    line_num +=1

f.close()

#print(f.readline())
