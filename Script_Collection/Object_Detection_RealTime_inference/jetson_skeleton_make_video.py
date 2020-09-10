#This script can be executed at /trt_pose/tasks/human_pose
#Of course it's python3

import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

with open('human_pose.json','r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.resnet18_baseline_att(num_parts, 2*num_links).cuda().eval()

MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))
print("load model finished")

WIDTH = 224
HEIGHT=224

data = torch.zeros((1,3,HEIGHT, WIDTH)).cuda()

model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)


OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)
print("torch saved finished")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))
print("load model to torch finished")


t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0/ (t1- t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229,0.224,0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:,None, None])
    return image[None, ...]

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


cap = cv2.VideoCapture('1_360p.mp4')
#cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('./demo_1_360p.mp4',cv2.VideoWriter_fourcc(*'mp4v'),30,(frame_width,frame_height))


if(cap.isOpened()==False):
    print("Error opening video stream or file")


while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        resize =cv2.resize(frame,(224,224),interpolation =cv2.INTER_LINEAR)
        data = preprocess(resize)
        t2 = time.time()
        cmap,paf = model_trt(data)
        cmap,paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects,peaks = parse_objects(cmap,paf)
        t3 = time.time()
        print("Time:::::    ",(1.0/(t3-t2)))
        #print(peaks)
        draw_objects(resize,counts, objects,peaks)
        resized = cv2.resize(resize,(frame_width,frame_height),interpolation = cv2.INTER_LINEAR)
        #1280 760
        out.write(resized)
        #cv2.imshow('Frame',resized)

        #if cv2.waitKey(25) & 0xFF == ord('q'):
            #break

    else:
        break
cap.release()
out.release()

#cv2.destroyAllWindows()











