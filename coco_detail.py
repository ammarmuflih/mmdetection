# Import modul COCO
from pycocotools.coco import COCO

# Inisialisasi COCO API
coco = COCO("/home/ammar/Documents/mmdetection/data/coco2/annotations/instances_val2017.json")
jumlah = 0
classes = coco.dataset['categories']
# Mencetak daftar kelas yang terdapat dalam dataset COCO
for c in classes:
    jumlah = jumlah+1
    print(c['name'])

print('Jumlah = ',jumlah)