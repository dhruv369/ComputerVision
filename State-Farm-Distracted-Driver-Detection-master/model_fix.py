

'''
Dhruv vyas

In Progress state-farm-distracted-driver-detection



'''
# Importing all necessary libraries
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K


train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples =400
nb_validation_samples = 100
epochs = 10
batch_size = 16
img_width = 224
img_height = 224


def generator_fun(model):
    nb_train_samples = 20399
    nb_validation_samples = 2025
    epochs = 1000
    batch_size = 128

    train_data_dir = "C:\\Users\\dhruv\\PycharmProjects\\CNN_Model1\\state-farm-distracted-driver-detection\\imgs\\train"
    validation_data_dir = "C:\\Users\\dhruv\\PycharmProjects\\CNN_Model1\\state-farm-distracted-driver-detection\\imgs\\validation"
    #model_path = ''
    model_name = 'model_1_saved.h5'

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)


    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    checkpoint_file = 'filepath="weights_improvement_1.hdf5'
    filepath ='C:\\Users\\dhruv\\PycharmProjects\\CNN_Model1\\ModelCheckpoint'

    model_check = keras.callbacks.ModelCheckpoint(filepath+checkpoint_file, monitor='val_loss', verbose=0, save_best_only=True,
                                    save_weights_only=False, mode='auto', period=1)

    dir_log = 'C:\\Users\\dhruv\\PycharmProjects\\CNN_Model1\\logs'
    tensor_board_log = keras.callbacks.TensorBoard(log_dir=dir_log, histogram_freq=0,
                                write_graph=True, write_images=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[tensor_board_log,model_check]
         )

    model.save_weights('weights_improvement_1.h5')


def model_def():

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('sigmoid'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])

    generator_fun(model)

if __name__ == '__main__':
    print("start")
    model_def()
    print("end")