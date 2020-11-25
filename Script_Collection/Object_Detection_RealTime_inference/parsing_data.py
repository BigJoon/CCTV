#해당 경로에 해당하는 xml 파일들을 읽어온후 obstacle_classes 리스트에 존재하는 tag들이 포함된 xml 파일들을 뽑아서 "trainval.txt"로 만들어주는 꿀스크립트
# Made by DR.JHSong

import os
from xml.etree.ElementTree import parse

#여기엔 원본 데이터에서 VOC2007 까지의 경로를 넣어주면 될듯?
dataset_path = '/hdd/TMP_D/Resized_Data_Voc_For_Song/VOCdevkit/VOC2007'
trainval_txt_path = os.path.join(dataset_path, 'ImageSets/Main/trainval.txt')
test_txt_path = os.path.join(dataset_path, 'ImageSets/Main/test.txt')
annotations_dir_path = os.path.join(dataset_path, 'Annotations')
obstacle_classes = ['person', 'bicycle', 'bus', 'car', 'motorcycle_movable', 'movable_signage', 'trunk', 'chair']
# obstacle_classes = ['person', 'bicycle', 'bus', 'car', 'carrier',
#     'motorcycle', 'movable_signage', 'truck', 'bollard', 'chair',
#     'potted_plant', 'table', 'tree_trunk', 'pole', 'fire_hydrant']

num_of_fold = 10
train_ratio = 9

origin_annotations_list = os.listdir(annotations_dir_path)
origin_annotations_len = len(origin_annotations_list)
annotations_list_train = origin_annotations_list[:int(origin_annotations_len/num_of_fold * train_ratio)]
annotations_list_test = origin_annotations_list[int(origin_annotations_len/num_of_fold * train_ratio)+1:]
annotations_lists = [annotations_list_train, annotations_list_test]
trainval_txt = open(trainval_txt_path, 'w')
test_txt = open(test_txt_path, 'w')
txt_list = [trainval_txt, test_txt]
print("INFO: # of Annotations: ", len(origin_annotations_list))
print("INFO: trainval.txt path: ", trainval_txt_path)

annotations_count = 0
for annotations_list, txt in zip (annotations_lists, txt_list) :
    for idx, annotation_name in enumerate(annotations_list):
        print("\r[{}/{}]".format(idx, len(annotations_list)), end='')
        try :
            annotation = parse(os.path.join(annotations_dir_path, annotation_name))
            objects = annotation.findall("object")
            names = [obj.findtext("name") for obj in objects]
            is_object = False
            if len(objects) == 0 :
                print(annotation_name)
            for name in names:
                for obstacle_class in obstacle_classes :
                    if name == obstacle_class :
                        is_object = True
            for object in objects:
                bbox = object.find('bndbox')
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                if x2 == 0 or y2 == 0 :
                    print(annotation_name)

            if is_object == True:
                txt.write(annotation_name.replace(".xml", "") + "\n")
                annotations_count += 1
            else :
                print("\t" + annotation_name)
        except :
            pass
    print()
            # print(annotation_name)
print()
trainval_txt.close()
test_txt.close()
print("INFO: train annotation count: ", annotations_count)
