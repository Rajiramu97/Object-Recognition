import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time

with open("conf/conf.json") as f:
	config=json.load(f)

model_name=config["model"]
weights=config["weights"]
include_top=config["include_top"]
train_path=config["train_path"]
features_path=config["features_path"]
labels_path=config["labels_path"]
test_size=config["test_size"]
results=config["results"]
model_path=config["model_path"]

print("[status] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start=time.time()


if model_name=="vgg16":
	base_model=VGG16(weights=weights)
	model=Model(inputs=base_model.input,outputs=base_model.get_layer('fc1').output)
	image_size=(224,224)
	
else:
	base_model=None


print("[info] successfully loaded base model and model.....")
train_labels=sorted(os.listdir(train_path))


print("[info] encoding labels....")
le=LabelEncoder()
le.fit([tl for tl in train_labels])


features=[]
labels=[]


count=1
for i,label in enumerate(train_labels):
	cur_path=train_path+"/"+label
	count=1
	for image_path in glob.glob(cur_path+"/*.jpg"):
		img=image.load_img(image_path,target_size=image_size)
		x=image.img_to_array(img)
		x=np.expand_dims(x,axis=0)
		x=preprocess_input(x)
		feature=model.predict(x)
		flat=feature.flatten()
		features.append(flat)
		labels.append(label)
		print("[info] processed-"+str(count))
		count=count+1
	print("[info] completed label-"+label)


le=LabelEncoder()
le_labels=le.fit_transform(labels)
print("[status] training labels:{}".format(le_labels))
print("[status] training labels shape:{}".format(le_labels.shape))

h5f_data=h5py.File(features_path,'w')
h5f_data.create_dataset('dataset_1',data=np.array(features))


h5f_label=h5py.File(labels_path,'w')
h5f_label.create_dataset('dataset_1',data=np.array(le_labels))


h5f_data.close()
h5f_label.close()

model_json=model.to_json()
with open(model_path+str(test_size)+".json","w") as json_file:
	json_file.write(model_json)
model.save_weights(model_path+str(test_size)+".h5")
print("[status] saved model and weights to disk....")
print("[status] saved labels and features")
end=time.time()
print("[status] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

