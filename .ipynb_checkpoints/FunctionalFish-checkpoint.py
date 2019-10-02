
from wandb.keras import WandbCallback
from pathlib import Path
import json
import wandb
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.layers import BatchNormalization, Dense, RepeatVector
from keras.models import Model

from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

run = wandb.init()

config = run.config

bb_params = ['height', 'width', 'x', 'y']
config.width = 224 # or 299
config.height = 224 # or 299


def convert_bb(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (config.width / size[0])
    conv_y = (config.height / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def to_plot(img):
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)

def plotfish(img):
    plt.imshow(to_plot(img))
    
def show_bb(i):
    bb = val_bbox[i]
    plotfish(val[i])
    plt.gca().add_patch(create_rect(bb))
    
    
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft', 'NoF']
bb_json = {}
path = "kagfish/bbox"
for c in anno_classes:
    if c == 'other': continue # no annotation file for "other" class
    j = json.load(open('{}/{}_labels.json'.format(path, c), 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations'])>0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height']*x['width'])[-1]
bb_json['img_04908.jpg']


img_width, img_height = 224,224

input_shape = (img_width,img_height,3)

train_data_dir = '../SoLong/kagfish/train'

config.n_train_samples = 200
config.n_validation_samples = 200
config.epochs = 10
batch_size = 64

conv_base = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# conv_base.trainable = False

# set_trainable = False
# for layer in conv_base.layers:
#     print(">>", layer.name,layer.trainable)
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     #if layer.name == 'block4_conv1':
#      #   set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#     print("OUT", layer.name,layer.trainable)



# this is the augmentation configuration we will use for validation:
# only rescaling

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (config.width, config.height),
        batch_size = batch_size,
        shuffle = True,
        subset = "training",
        classes = anno_classes,
        class_mode = 'categorical')


validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (config.width, config.height),
        batch_size = batch_size,
        shuffle = True,
        subset = "validation",
        classes = anno_classes,
        class_mode = 'categorical')



 # Not sure but we took 224x224x3
# entry = Input# Not sure?
entry = Input(shape=(224, 224, 3,), name="entry")
x = VGG16(conv_base)(entry)

print(x)

#x = Convolution2D(32, 3, 3, activation = 'relu')(x)
#x = MaxPooling2D(MaxPooling2D((2,2), strides=(2,2)))(x)
#x = Flatten()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(inputs=entry, outputs=predictions)

print(model.summary())

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

best_model_file = "./weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

model.summary()

debugger = input()

model.fit_generator(
        train_generator,
        samples_per_epoch = config.n_train_samples,
        nb_epoch = config.epochs,
        validation_data = validation_generator,
        nb_val_samples = config.n_validation_samples,
        callbacks = [WandbCallback()])


sgd = SGD(lr = 1e-4)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])
