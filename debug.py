from PIL import Image
from django.conf import settings
import numpy as np
import copy as cp
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "imgsearch.settings")
img = Image.open("C://Users/Kudo/Documents/Kuliah/Semester 7/Data Mining/Pertemuan 9 (Multiband Image Clustering)/Dataset/GIF/LM21260651979278AAA05.gif")
img = np.array(img)
# coba = cp.deepcopy(img)
# img2 = coba.ravel()
# img2 = np.reshape(img2, (-1,32))

a = [[1,2,3], [2,3,4], [3,6,8], [7,2,5]]
c = ['usaha', 'sama', 'kamu']
b = np.array(a)
b = np.reshape(b, (2,6))
b = np.reshape(b, (2,2,3))

mp = b.max(axis=1)
mp = np.array(mp).tolist()

w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[256, 256] = [255, 0, 0]
img = Image.fromarray(data, 'RGB')
img.save(settings.STATIC_DIR + '\my.png')
img.show()

print("HAHA")