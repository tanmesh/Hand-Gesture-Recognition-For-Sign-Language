# Part 1 - Building the CNN
# importing the Keras libraries and packages
from keras import optimizers
from keras.layers import Conv2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.applications.resnet50 import ResNet50

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Initialing the CNN
classifier = Sequential()

# Step 1 - Convolution Layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding second convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding 3rd Concolution Layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(32, activation='softmax'))

# Compiling The CNN
classifier.compile(
    optimizer=optimizers.SGD(lr=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Part 2 Fittting the CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

csv_logger = CSVLogger("model_history_log.csv", append=True)

model = classifier.fit_generator(
    training_set,
    steps_per_epoch=800,
    epochs=25,
    validation_data=test_set,
    validation_steps=6500,
    callbacks=[csv_logger]
)

'''#Saving the model
import h5py
classifier.save('Trained_model.h5')'''

'''#Saving the model
import h5py
classifier.save('Trained_model_alpanumeric.h5')'''

print(model.history.keys())
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy2.png')

plt.clf()
# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss2.png')
