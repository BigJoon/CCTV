#이 프로그램은 path 경로에 있는 파일을 <folder>,<path>, <source>, <database>,<segmented>,<pose>,<truncated> 태그들을 지워줌.
#누워있는 사람 데이터셋에 문제가 많았음. 정상적인 tfrecord화를 시키기 위하여 기존 obstacle dataset과 형식을 맞춰주기 위함임.

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
        
        for test in roote.iter('Annotation'):
            for stuff in test.findall('folder'):
                test.remove(stuff)
            for stuff in test.findall('path'):
                test.remove(stuff)
            for stuff in test.findall('source'):
                test.remove(stuff)
            for stuff in test.findall('segmented'):
                test.remove(stuff)

            for stuff in test.findall('object'):
                for tmp in stuff.findall('pose'):
                    stuff.remove(tmp)
            for stuff in test.findall('object'):
                for tmp in stuff.findall('truncated'):
                    stuff.remove(tmp)

        #print(roote.tag)
        #roote.tag = "Annotation"

        #tree.write(path+'test.xml')
        tree.write(path+fname)

        """
        for tags in root.findall("annotation"):
            path_trash = tags.find('path')
            
            print(path_trash)
        """
