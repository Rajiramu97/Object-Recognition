from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

with open("conf/conf.json") as f:
	config=json.load(f)

test_size	= config["test_size"]
seed	= config["seed"]
features_path	= config["features_path"]
labels_path	= config["labels_path"]
results	= config["results"]
classifier_path	= config["classifier_path"]
train_path	= config["train_path"]
num_classes	= config["num_classes"]
classifier_path	= config["classifier_path"]

h5f_data=h5py.File(features_path,'r')
h5f_label=h5py.File(labels_path,'r')

features_string=h5f_data['dataset_1']
labels_string =h5f_label['dataset_1']

features=np.array(features_string)
labels=np.array(labels_string)

h5f_data.close()
h5f_label.close()


print("[INFO] features shape:{}".format(features.shape))
print("[INFO] labels shape:{}".format(labels.shape))

print("[info] training started...")

(trainData,testData,trainLabels,testLabels)=train_test_split(np.array(features),np.array(labels),test_size=test_size,random_state=seed)

print("[info] splitted train and test data..")
print("[info] train data : {}".format(trainData.shape))
print("[info] test data : {}".format(testData.shape))
print("[info] train labels : {}".format(trainLabels.shape))
print("[info] test labels : {}".format(testLabels.shape))

print("[info] creating model...")
model=LogisticRegression(random_state=seed)
model.fit(trainData,trainLabels)

print("[info] evaluating model..")
f=open(results,"w")
rank=0


for(label, features) in zip(testLabels, testData):

  predictions=model.predict_proba(np.atleast_2d(features))[0]
  predictions=np.argsort(predictions)[::-1][:5]

  if label==predictions[0]:
  	rank=rank+1

rank=(rank/float(len(testLabels)))*100
 

f.write("Rank: {:.2f}%\n".format(rank))
  

preds = model.predict(testData)

f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

print("[INFO] saving model...")
pickle.dump(model,open(classifier_path,'wb'))


print("[INFO] confusion matrix")



labels = sorted(list(os.listdir(train_path)))


cm=confusion_matrix(testLabels,preds)

print(cm)
print(sns.heatmap(cm,annot=True, cmap="Set2"))
plt.show()












