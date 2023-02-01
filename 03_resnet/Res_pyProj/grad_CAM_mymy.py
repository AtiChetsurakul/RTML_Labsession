from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from model_calling import *
from pytorch_grad_cam.utils.image import deprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet50
import torch
import cv2
import numpy as np
import os
print(os.listdir('../cam'))
img_name = os.listdir('../cam')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = ResSENet18()

model.load_state_dict(torch.load(
    '/root/keep_lab/RTML_Labsession/03_resnet/Res_pyProj/Result/SEresnet18_bestsofar.pth'))

for index, i in enumerate(img_name):
    # model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    rgb_img = cv2.imread(f"../cam/{i}", 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    # print(rgb_img.shape)
    # print(rgb_img.shape)
    # break
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # print(input_tensor.size())
    # break
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    targets = [ClassifierOutputTarget(0), ClassifierOutputTarget(1), ClassifierOutputTarget(
        2), ClassifierOutputTarget(3), ClassifierOutputTarget(4), ClassifierOutputTarget(5), ClassifierOutputTarget(6), ClassifierOutputTarget(
        7), ClassifierOutputTarget(8), ClassifierOutputTarget(9)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite(f'../cam_out/ex{index}_cam.jpg', cam_image)
    # cv2.imwrite(f'../cam_out/ex{index}_gb.jpg', gb)
    # cv2.imwrite(f'../cam_out/ex{index}_cam_gb.jpg', cam_gb)
    print(f'{index}/{len(img_name)}')
