'''
# CLASSIFY IMAGES. CLASSIFICA L'EXEMPLE DE 3 SCENES AMB ELS SEGÜENTS ALGORITMES DE CLASSIFICACIO
# K-Nearest-Neighbour
# Naive Bayes Classificator
# Regressió Logística
# Support Vector Machine
# Arbres de Decisió
# Random Forests
# Multi-Layer Perceptron (proto - Xarxa Neuronal)
# Primer fa data extraction (treu característiques de cada píxel per a poder tractar sobre dades numèriques)
'''

# imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os
from scipy import stats


def extract_color_stats(image):
	# Parteix cada pixel en RGB i crea un vector amb 6 valors
	# La mitja i la desviacio estandard de cadascun dels 3 canals
	# També la mediana
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B), np.median(R), np.median(G), np.median(B)]

	return features

#Seleccionar el model
print("Select model: knn, naive_bayes, logit, svm, decision_tree, random_forest, mlp")
modelsel = input()
dataset = "train"

# Defineix el diccionari de models.
# Cada model cridarà a una funció diferent del sklearn
# fa l'equivalent a un switch
models = {
	"knn": KNeighborsClassifier(n_neighbors=15),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
	"svm": SVC(kernel="linear"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators=20),
	"mlp": MLPClassifier()
}

# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("[INFO] extracting image features...")
imagePaths = paths.list_images(dataset)
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, compute color channel
	# statistics, and then update our data list
	image = Image.open(imagePath)
	# extreu característiques dels colors. Valors RBG de cada pixel, contrast, sobretot intensitat, etc etc etc.
	features = extract_color_stats(image)
	data.append(features)

	#agafa la label. La label es el nom de la carpeta a on està la imatge. Vas al path. vas dos passos enrere (del path), per trobar el nomd e la carpeta
	#i ho poses com a label.
	label = imagePath.split(os.path.sep)[-1][0]
	labels.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform a training and testing split, using 75% of the data for
# training and 25% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25)

# train the model
print("[INFO] using '{}' model".format(modelsel))
model = models[modelsel]
model.fit(trainX, trainY)

# make predictions on our data and show a classification report


# Dates de train
print("[INFO] evaluating train cases...")
predictions = model.predict(trainX)
print(classification_report(trainY, predictions,
	target_names=le.classes_))

# Dates de test
print("[INFO] evaluating test cases...")
predictions = model.predict(testX)
print(classification_report(testY, predictions,
	target_names=le.classes_))
