import torch
import torchvision
from torchvision.datasets import CocoDetection

# unduh dataset Coco
coco = CocoDetection('/home/ammar/fiftyone/coco-2017/train/labels.json')

# hapus kelas yang tidak diinginkan
coco.remove_categories(["chair"])


# from pycocotools.coco import COCO

# coco = COCO('/home/ammar/fiftyone/coco-2017/train/labels.json')
# classes = coco.dataset['categories']
# selectec_classes = ['person','car']

# coco.dataset['categories'] = [c for c in classes if c['name'] in selectec_classes]

# coco.save()