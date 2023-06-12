from tensorflow.keras.applications import DenseNet121

from numpy.random import  seed 
import  time 
from keras import applications
from keras import callbacks
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from gc import callbacks
from keras.callbacks import CSVLogger
from keras.layers import Dense
from keras import optimizers
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K




t = time.time()

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'Data/train/'
val_data_dir = 'Data/val/'
test_data_dir = 'Data/test/'


##Data augmentation preprocessing
# used to rescale the pixel values from [0, 255] to [0, 1] interval


datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.00, 
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)



batch_size = 32

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')


def build_model():
    base_model = densenet.DenseNet121(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False, pooling='avg')
    for layer in base_model.layers:
      layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

model = build_model()
model.summary()


#Describing the optimization scheme and the cost function 
model.compile(loss='sparse_categorical_crossentropy',  optimizer='adam', metrics=['accuracy'])



#Training
epochs = 5
train_samples = 5216
validation_samples = 624
test_samples = 16


csv_logger = CSVLogger('log1.csv', append=True, separator=';')


#Fitting the model on the training data
model.fit_generator(train_generator, steps_per_epoch=train_samples // batch_size, epochs=epochs,validation_data=validation_generator, validation_steps=validation_samples// batch_size, callbacks = csv_logger)


model.save('COVIDDenseNet121_5_epochs.h5')

#Evaluating on validation set for Computing loss and accuracy :

model.evaluate_generator(validation_generator, validation_samples)

