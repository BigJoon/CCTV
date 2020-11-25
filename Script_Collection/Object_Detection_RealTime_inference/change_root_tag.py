#이 프로그램은 path 경로에 있는 파일을 읽어와서 annotation을 Annotation으로 바꿔준다.
#용도는 누워있는 사람 데이터셋 헤드 태그가 소문자로 되어 있어서 그걸 일괄적으로 바꿔주기 위함이였다.
#Made by YJHwang
#path
import os
import xml.etree.ElementTree as ET

#XML 파일 들이 들어있는 경로를 "path"에 넣으면 됨.
#path = "/hdd/TMP_D/temporal/sample/"
path = "/hdd/Fresh_Data/Fall_Down_Data/Annotations/"
for root, dirs, files in os.walk(path):
    for fname in files:
        #full_fname = os.path.join(root,fname)
        tree = ET.parse(path+fname)
        roote = tree.getroot()

        print(roote.tag)
        roote.tag = "Annotation"

        tree.write(path+fname)

        """
        for tags in root.findall("annotation"):
            path_trash = tags.find('path')
            
            print(path_trash)
        """
