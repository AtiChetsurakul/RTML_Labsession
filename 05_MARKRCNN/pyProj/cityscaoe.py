from torchvision.datasets import Cityscapes

dataset = Cityscapes('/root/Datasets/Cityscapes', split='train', mode='fine',
                     target_type='instance')

img, smnt = dataset[0]

print(smnt)