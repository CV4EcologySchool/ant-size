import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
from glob import glob

data = "s3://ant-size/data/"

file_list = []
for root, dirs, files in os.walk(data):
    for file in files:
        if file.endswith('.JPG'):
            file_list.append(file)

date = []
id1 = []
id2 = []
object = []
camera = []
mass = []
fn = []

for f in file_list:
    fn.append(f)   
    f = os.path.splitext(f)[0]   
    date.append(f.split('_')[0])
    id1.append(f.split('_')[1])
    id2.append(f.split('_')[2])
    object.append(f.split('_')[3])
    camera.append(f.split('_')[4])
    mass.append(float(f.split('_')[6]))

print('Data Information\n')
print('Images: '+str(len(file_list)))
print('Weight classes: '+str(len(df.mass.unique())))
print('Weight range: '+str(df.mass.min()*1000)+'mg to '+str(df.mass.max()*1000)+'mg')
print('Cameras used: '+str(len(df.camera.unique())))
print('Dates filmed: '+str(len(df.date.unique())))

images = []

# select some random images
for i in random.sample(range(0, len(files)), 10):
  images.append(mpimg.imread(str(files[i])))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.savefig('ant_sample.png')