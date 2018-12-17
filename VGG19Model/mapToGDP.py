import cv2
import numpy as np
import os
import pandas as pd

from random import shuffle
from tqdm import tqdm
np.random.seed(5)
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))

'''
# Hyperparameter
'''

IMG_SIZE = 256
LR = 0.001

'''
# set train indices and test indices in random
'''
dir = d + '/ImageSet/RoadMap/'
labelSet = pd.read_csv(d + '\Data\PopulationLabelC.csv').values
indices = np.random.choice(3953,3953,replace=False)
train_indices = indices[400:]
test_indices = indices[:400]

MODEL_NAME = 'GdpModel-{}-{}.model'.format(LR,'10conv-basic')

def label_img(img):
    labelidx = img.split('.')[0]
    res = [0 for _ in range(10)]
    res[labelSet[int(labelidx) - 1][1]] = 1
    return res

def change_name_to_digit():
    for img in tqdm(os.listdir(dir)):
        labelidx = img.split('.')[0]


# memo: 1006,change 'i', 1036 's',
def process_data():
    cur_data = []
    count = 0
    id = 0
    cv_size = lambda img: tuple(img.shape[1::-1])
    for img in tqdm(os.listdir(dir)):
        label = label_img(img)
        path = os.path.join(dir,img)
        # In this place I use open and change to byte to read some special character
        # Also Facing Memory Error in this place.
        stream = open(path, "rb")
        bytes = bytearray(stream.read())
        numpyarray = np.asarray(bytes, dtype=np.uint8)
        img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED) # decode the byte image
        #img = cv2.imread(path,3)
        #cv2.imwrite('kk.jpg',img)
        # !! resize
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        # for one hot encoding
        cur_data.append([np.array(img),np.array(label)])
        count += 1
        if count == 400:
            count = 0
            shuffle(cur_data)
            np.save(d + '/Data/cur_data' + str(id) +'.npy',cur_data)
            cur_data = []
            id += 1
    shuffle(cur_data)
    np.save(d + '/Data/cur_data' + str(id) + '.npy', cur_data)

#process_data()

Data = np.empty((0,2))
for i in range(10):
    print(i)
    cur_data = np.load(d + '/Data/cur_data' + str(i) +'.npy')
    Data = np.concatenate((Data,cur_data),axis=0)
    print(Data.shape)

# Out of memory problem with 1280x1280
train_data = Data[train_indices]
test_data = Data[test_indices]
print(train_data.shape,test_data.shape)
del Data



import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d,conv_3d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

# adding
import tensorflow as tf
tf.reset_default_graph()




network = input_data(shape=[None,IMG_SIZE,IMG_SIZE,3])


network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = max_pool_2d(network, 2, strides=2)

network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')

network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_vgg',
                    max_checkpoints=1, tensorboard_verbose=0)

'''
convnet = input_data(shape=[None,IMG_SIZE,IMG_SIZE,3],name = 'input')

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,64,5,activation='relu')
convnet = conv_2d(convnet,128,5,activation='relu')
convnet = max_pool_2d(convnet,5)

convnet = conv_2d(convnet,32,5,activation='relu')
convnet = conv_2d(convnet,64,5,activation='relu')
convnet = max_pool_2d(convnet,5)


convnet = fully_connected(convnet,1024,activation = 'relu')
convnet = dropout(convnet,0.8)

convnet = fully_connected(convnet,10,activation='softmax')
convnet = regression(convnet,optimizer = 'adam',learning_rate = LR,loss = 'categorical_crossentropy',name = 'targets')

model = tflearn.DNN(convnet,tensorboard_dir = 'log')
'''
# simple split
X = np.array([i[0] for i in train_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train_data]

test_x = np.array([i[0] for i in test_data]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test_data]



if os.path.exists('{}.meta'.format(MODEL_NAME)) and 1==1:
    model.load(MODEL_NAME)
else:
    print(X.shape)
    model.fit(X, Y, n_epoch=500, shuffle=True,
              show_metric=True, batch_size=32, snapshot_step=500,
              snapshot_epoch=False, run_id='vgg_oxflowers17')
    model.save(MODEL_NAME)
    pass


import matplotlib.pyplot as plt

fig = plt.figure()

result = []
correct = 0
for num,data in enumerate(test_data):
    gdp = data[1]
    img_data = data[0]

    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,3)
    model_out = model.predict([data])[0]
    print(np.argmax(model_out),np.argmax(gdp))
    result.append([np.argmax(model_out),np.argmax(gdp)])
    if np.argmax(model_out) == np.argmax(gdp):
        correct += 1
result = np.asarray(result)
result = pd.DataFrame(result)
result.to_csv("testCP.csv")
print('acc',correct/len(test_data))

