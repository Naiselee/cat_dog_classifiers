# Classificador de 3scenes mitjançant Deep Learning. Fa servir xarxes neuronals convolucionades

# import the necessary packages
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
# from keras.optimizers import Adam
# from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os

dataset = "train"

# Agafa totes les imatges del dataset i les classifica
print("[INFO] loading images...")
imagePaths = paths.list_images(dataset)
data = []
labels = []
# bucle sobre les imatges
for imagePath in imagePaths:
    # carrega la imatge, la reescala a 32x32. Compte. Això es perque vagi més rapid pero evidentment perjudicarà la prediccio.
    # remapeja l'intensistat del pixel del minim al maxim (per a que quedi mes diferenciada)

    image = Image.open(imagePath)
    image = np.array(image.resize((64, 64))) / 255.0
    data.append(image)

    # Treu l'etiqueta del path (es a dir, l'etiqueta es el nom de la carpeta)
    label = imagePath.split(os.path.sep)[-1][0]
    labels.append(label)

# codifica les etiquetes a enters
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Carrega el dataset i el divideix en dos datasets aleatoriament
# Fa servir 3/4 per training i 1/4 per avaluació
(trainX, testX, trainY, testY) = train_test_split(np.array(data),
                                                  np.array(labels), test_size=0.50)

# ACTIVAR AQUESTES DUES LINIES EN CAS DE QUE EL CLASIFICADOR SIGUI BINARI (NOMES 2 OPCIONS)
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Que definim nosaltres:
# 1. Cuantes capes
# 2. Cuantes neurones per capa
# 3. Tipus de capa

# defineix la CNN
model = Sequential()
# Primera capa, de 8. aquesta es una capa d'input a on li entren els 32*32 pixels de l'imatge
# Tipus de capes que hi ha.
# Conv - Convolucional. Cuando estas asociado con el vecino. Hay de varias dimensiones
# Dense - Los datos estan relacionados pero no hay una relacion en que algunos estan mas relacionados que otros
# Dropout y Flatten
# Kerneling - el Kernel se aplica solo en las conv. crea una matriz alrededor del punto central. Es decir, en vez de analizar pixel a pixel, analiza grupos de 3x3 pixel
model.add(Conv2D(16, (3, 3), padding="same", input_shape=(64, 64, 3)))
# funcio d'activació. determina com de "segur" ha d'estar per a tornar 1.
model.add(Activation("elu"))
# El max pooling serveix per que la xarxa no quedi inmensa i tardi anys
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Segona capa, de 16, intermitja. el 3,3 es la mida del kernel (mesura cada pixel en relacio als pixels dels costats)
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Tercera capa, de 32. Sol ser bona idea que cada capa sigui el doble de l'anterior. Pero no sempre, aixi es el ML
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("5elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Cuarta capa. Why not
# model.add(Conv2D(64, (5, 5), padding="same"))
# model.add(Activation(keras.layers.LeakyReLU(alpha=0.3)))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Afegeix una capa final de 3, amb un classificador softmax (donarà output i serà amb percentatges de probabilitat)
model.add(Flatten())
model.add(Dense(2))
model.add(Activation("softmax"))

# Inicialitza l'optimitzador i model d'entrenament
print("[INFO] training network...")
# Adam es un bon optimitzador per CNN, pero no sempre el millor.
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3 / 50)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# Entrenament. Aqui has d'indicar cuantes "epoques" (generacions) vols entrenar l'algoritme.
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=8, batch_size=64)

# avaluació

##dates sobre el model de train
print("[INFO] evaluating network over training data...")
predictions = model.predict(trainX, batch_size=32)
print(classification_report(trainY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

##dates sobre el model de test
print("[INFO] evaluating network over validation data...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# mirar si son molt diferents i explicar overfitting i underfitting
