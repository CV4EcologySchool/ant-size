import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from glob import glob
import pandas as pd
import cv2


file_list = []
for root, dirs, files in os.walk('../data'):
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

data = {
    'id1': id1,
    'id2': id2,
    'date': date,
    'mass': mass,
    'object': object,
    'camera': camera,
    'filename': fn,
    }

df = pd.DataFrame(data)

print('Data Information\n')
print('Images: '+str(len(file_list)))
print('Weight classes: '+str(len(df.mass.unique())))
print('Weight range: '+str(df.mass.min()*1000)+'mg to '+str(df.mass.max()*1000)+'mg')
print('Cameras used: '+str(len(df.camera.unique())))
print('Dates filmed: '+str(len(df.date.unique())))


plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
plt.hist(df.mass*1000, bins = range(0, 30, 1))
plt.xlabel('Mass (mg)')
plt.ylabel('Number of images')
plt.title('Images per category')
#plt.grid(b=None)
plt.tight_layout()
plt.tick_params(axis='x', which='both', bottom=True, top=False)
plt.tick_params(axis='y', which='both', right=False, left=True)
plt.savefig('../fig1_hist.png')

# images = []

""" # select some random images
for i in random.sample(range(0, len(fn_list)), 10):
  images.append(mpimg.imread(fn_list[i]))

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(images):
    plt.subplot(len(images) / columns + 1, columns, i + 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.savefig('../ant_sample.png') """