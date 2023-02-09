import cv2
import torch

def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

from util import *
import darknet

blocks = darknet.parse_cfg("cfg/yolov3.cfg")
#print(darknet.create_modules(blocks))

model = darknet.Darknet("cfg/yolov3.cfg")
model.load_weights("yolov3.weights")
inp = get_test_input()
pred = model(inp, False)
#print (pred)
detection = write_results(pred.detach(), 0.5, 80, nms_conf = 0.4)
#print(detection)

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

num_classes = 80
classes = load_classes("data/coco.names")
#print(classes)
