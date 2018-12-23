from __future__ import print_function


from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input


from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import json
import pickle
import cv2
import pyttsx3


with open('conf/conf.json') as f:    
	config = json.load(f)


model_name 		= config["model"]
weights 		= config["weights"]
include_top 	= config["include_top"]
train_path 		= config["train_path"]
test_path 		= config["test_path"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
test_size 		= config["test_size"]
results 		= config["results"]
model_path 		= config["model_path"]
seed 			= config["seed"]
classifier_path = config["classifier_path"]


print ("[INFO] loading the classifier...")
classifier = pickle.load(open(classifier_path, 'rb'))


if model_name == "vgg16":
	base_model = VGG16(weights=weights)
	model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
	image_size = (224, 224)

else:
	base_model = None


train_labels = sorted(os.listdir(train_path))


test_images = sorted(os.listdir(test_path))


for image_path in test_images:
	path 		= test_path + "/" + image_path
	img 		= image.load_img(path, target_size=image_size)
	x 		= image.img_to_array(img)
	x 		= np.expand_dims(x, axis=0)
	x 		= preprocess_input(x)
	feature 	= model.predict(x)
	flat 		= feature.flatten()
	flat 		= np.expand_dims(flat, axis=0)
	preds 		= classifier.predict(flat)
	prediction 	= train_labels[preds[0]]
	
	
	print ("I think it is a " + train_labels[preds[0]])
	engine=pyttsx3.init()
	engine.say("I think it is a " + train_labels[preds[0]])
	engine.runAndWait()
	img_color = cv2.imread(path, 1)
	img_color = cv2.resize(img_color,(300,300))    
	cv2.putText(img_color, "I think it is a " + prediction, (140,445), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test", img_color)

	
	
	key = cv2.waitKey(0) & 0xFF
	if (key == ord('q')):
		cv2.destroyAllWindows()
