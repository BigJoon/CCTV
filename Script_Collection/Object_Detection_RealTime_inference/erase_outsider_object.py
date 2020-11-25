#이 프로그램은 path 경로에 있는 파일을 {person, bicycle, bus, car, motorcycle, truck} 을 제외한 나머지 태그들을 지워줌.
#{person, car, movable_signage, truck, bollard, chair, potted_plant,tree_trunk}
import os
import xml.etree.ElementTree as ET

#XML 파일 들이 들어있는 경로를 "path"에 넣으면 됨.
#path = "/hdd/TMP_D/temporal/sample/"
path = "/hdd/Fresh_Data/Class8_A/VOCdevkit/VOC2007/Annotations/"
for root, dirs, files in os.walk(path):
    for fname in files:
        #full_fname = os.path.join(root,fname)
        tree = ET.parse(path+fname)
        roote = tree.getroot()
        
        for test in roote.iter('Annotation'):
            """
            for stuff in test.findall('folder'):
                test.remove(stuff)
            for stuff in test.findall('path'):
                test.remove(stuff)
            for stuff in test.findall('source'):
                test.remove(stuff)
            for stuff in test.findall('segmented'):
                test.remove(stuff)
            """
            #이 문제는 element로 찾아서 지워줘야함.
            for stuff in test.findall('object'):
                #object에서 바로 아래 태그가 name인데 그 element 가 traffic_sign이면 날려야함.
                if stuff.find("name").text != 'person' and stuff.find("name").text != 'bicycle' and  stuff.find("name").text!='bus' and stuff.find("name").text!='car' and stuff.find("name").text!= 'motorcycle' and stuff.find("name").text !='movable_signage' and stuff.find("name").text != 'truck' and stuff.find("name").text != 'chair':
                    test.remove(stuff)
                
                #비교를 이렇게 해선 안됨...
                #print(stuff.find("name").text)
                



        #tree.write(path+'test.xml')
        tree.write(path+fname)

        """
        for tags in root.findall("annotation"):
            path_trash = tags.find('path')
            
            print(path_trash)
        """
