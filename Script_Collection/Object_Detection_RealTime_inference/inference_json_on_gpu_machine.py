import numpy as np

import os
import pathlib

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pathlib

import json
from json import JSONEncoder
import numpy
from collections import OrderedDict

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile
tf.compat.v1.enable_eager_execution()


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)
    model_dir = pathlib.Path(model_dir) / "saved_model"

    # model = tf.saved_model.load(str(model_dir))
    model = tf.compat.v2.saved_model.load(str(model_dir), None)

    return model


# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = '/models/research/object_detection/data/mscoco_label_map.pbtxt'
PATH_TO_LABELS = '/hdd/test_training_output/fine_tuned_model/label_map.pbtxt'
PATH_TO_LABELS = '/hdd/Data/VOC2007/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print("----------------------\n------------------\n=\n")
print(category_index)
print("---------------------\n---------------\n-------------\n")

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/models/research/object_detection/test_images')
# TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS = [f'/hdd/Data/frames/{d}/{f}' for d in os.listdir('/hdd/Data/frames') for f in
                    os.listdir(f'/hdd/Data/frames/{d}') if f.endswith('.jpg')]
print("---------------------------------imageimgae-------------")
print(TEST_IMAGE_PATHS)
print("---------------------------------endendendendendendendend-------------")
# tf.saved_model.load() -> deprecated
# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# detection_model = load_model(model_name)
model_dir = '/hdd/test_training_output/fine_tuned_1.12_150000/saved_model/'
detection_model = tf.compat.v2.saved_model.load(str(model_dir), None)

#print(detection_model.signatures['serving_default'].inputs)
#print(detection_model.signatures['serving_default'].output_dtypes)
#print(detection_model.signatures['serving_default'].output_shapes)

for image_path in TEST_IMAGE_PATHS:
    #ADDING********************
    test_annos=dict()
    big_result=[]
    result = []
    
    #result 안에 label이랑 position이 들어가는데
    #label안에 ds가 들어감

    #label.append(ds)

    #result.append(label)
    #result.append(position)
    #big_result.append(result)
    




    #im =(Image.open(image_path))
    #tmp_w,tmp_h=im.size
    #print("SSSSSSSSSSSSSSSSSSSSSSSSSIZEEEEEEEEEEEEEE", tmp_w,tmp_h)
    #image_np = np.array(im)
    image_np = np.array(Image.open(image_path))
    image = np.asarray(image_np)

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = detection_model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    #print("numnumnumnum:",num_detections)
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        #print(output_dict['detection_masks_reframed'])

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
    
    im = Image.fromarray(image_np)
    #print("CCCCCCATEGORYYYY:",category_index)
    #print("BOXBOXBOXBOXBOX:", output_dict['detection_boxes'][0][0])
    """
    [i][0]은 ymin
    [i][1]은 xmin
    [i][2]는 ymax
    [i][3]은 xmax



    """
    for i in range(0,num_detections):
        ymin = 300*float(output_dict['detection_boxes'][i][0])
        xmin = 300*float(output_dict['detection_boxes'][i][1])
        ymax = 300*float(output_dict['detection_boxes'][i][2])
        xmax = 300*float(output_dict['detection_boxes'][i][3])
        label = []
        ds = {
                "label":[
                {
                    "description": category_index[int(output_dict['detection_classes'][i])]['name'],
                    "score": float(output_dict['detection_scores'][i]*100)

                }
                ],
                "position":{
                    "h": 1.2*(ymax-ymin),
                    "w": 1.6*(xmax-xmin),
                    "x": 1.6*(xmin+(xmax-xmin)/2),
                    "y": 1.2*(ymin+(ymax-ymin)/2)
                    }
             }
        result.append(ds)
        tmp = {
                "detection_result": result
                }
        big_result.append(tmp)

    test_annos={"image_path": image_path,
                "modules": "mobilenet-ssd-v2",
                "results": big_result
               }
    print("Image Path::::::::", str(image_path).split('/')[-2:])
    d,f = str(image_path).split('/')[-2:]
    g = f.split('.')
    print("GGGG",g[0])

    fd = open("/workspace/yj/frames_json_1.12/"+g[0]+'.json','w')
    print(json.dump(test_annos,fd, indent=4, cls=NumpyArrayEncoder))
    fd.close()

    print("Image_PATH ",image_path[17:])
    print("detection boxes",output_dict['detection_boxes'])
    print("detection_classes",output_dict['detection_classes'])
    print("detection_scores",output_dict['detection_scores'])
    print("===================================================")
    #break

    #박스 그릴때 이미지 가로세로 길이 읽어와서 곱해준 값으로 넣어줘야할듯?
    # d,f = str(image_path).split('/')[-2:]
    # if not os.path.exists(f'result/{d}'):
    #     os.makedirs(f'result/{d}')
    # im.save(f'result/{d}/{f}')
