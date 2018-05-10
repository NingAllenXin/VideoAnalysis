import cv2
import numpy as np
import glob
import os
import matplotlib as mpl
import imutils

mpl.use('TkAgg')
import matplotlib.pyplot as plt

def feature_surf(fn, extractor, detector):
    img = cv2.imread(fn, 0)
    return extractor.compute(img, detector.detect(img))[1]

def feature_bow(img, extractor_bow, detector):
    return extractor_bow.compute(img, detector.detect(img))

pos_im_path = "train/image/positive"
neg_im_path = "train/image/negative"

detector = cv2.xfeatures2d.SURF_create()
extractor = cv2.xfeatures2d.SURF_create()

flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})
#FLANN stands for Fast Library for Approximate Nearest Neighbors. It contains a collection of algorithms optimized for fast nearest neighbor search in large datasets and for high dimensional features

bow_kmeans_trainer = cv2.BOWKMeansTrainer(13)
bow_extract = cv2.BOWImgDescriptorExtractor(extractor, matcher)

for im_path in glob.glob(os.path.join(pos_im_path, "*jpg")):
    bow_kmeans_trainer.add(feature_surf(im_path, extractor, detector))

for im_path in glob.glob(os.path.join(neg_im_path, "*jpg")):
    bow_kmeans_trainer.add(feature_surf(im_path, extractor, detector))

vocabulary = bow_kmeans_trainer.cluster()
bow_extract.setVocabulary(vocabulary)

traindata, trainlabels = [], []

for im_path in glob.glob(os.path.join(pos_im_path, "*jpg")):
    traindata.extend(feature_bow(cv2.imread(im_path, 0), bow_extract, detector))
    trainlabels.append(1)

for im_path in glob.glob(os.path.join(neg_im_path, "*jpg")):
    traindata.extend(feature_bow(cv2.imread(im_path, 0), bow_extract, detector))
    trainlabels.append(-1)

svm = cv2.ml.SVM_create()
svm.trainAuto(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

svm.save("Temoc.xml")

s = cv2.ml.SVM_load("Temoc.xml")

camera = cv2.VideoCapture(0)

while True:
    success, frame = camera.read()
    frame = imutils.resize(frame, width=600)
    test = feature_bow(frame, bow_extract, detector)
    a, result = s.predict(test)
    _, res = s.predict(test, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    if result[0][0] == 1 and res[0][0] > -0.3:
        cv2.putText(frame, "Temoc", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
    cv2.imshow("Temoc", frame)
    key = cv2.waitKey(33) & 0xFF
    # Press 'q' to quit
    if key == ord("q"):
        break
