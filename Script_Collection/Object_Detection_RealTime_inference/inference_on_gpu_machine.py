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
#PATH_TO_LABELS = '/hdd/test_training_output/fine_tuned_model/label_map.pbtxt'
PATH_TO_LABELS = '/hdd/Data/VOC2007/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('/models/research/object_detection/test_images')
# TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS = [f'/hdd/Data/frames/{d}/{f}' for d in os.listdir('/hdd/Data/frames') for f in
                    os.listdir(f'/hdd/Data/frames/{d}') if f.endswith('.jpg')]
print(TEST_IMAGE_PATHS)

# tf.saved_model.load() -> deprecated
# model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
# detection_model = load_model(model_name)
#model_dir = '/hdd/test_training_output/fine_tuned_model/saved_model'
model_dir = '/hdd/test_training_output/8_25_exported/saved_model'
detection_model = tf.compat.v2.saved_model.load(str(model_dir), None)

print(detection_model.signatures['serving_default'].inputs)
print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)

for image_path in TEST_IMAGE_PATHS:
    image_np = np.array(Image.open(image_path))
    image = np.asarray(image_np)

    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    model_fn = detection_model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        print(output_dict['detection_masks_reframed'])

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


    d,f = str(image_path).split('/')[-2:]
    if not os.path.exists(f'result_tmp/{d}'):
        os.makedirs(f'result_tmp/{d}')
    im.save(f'result_tmp/{d}/{f}')
