#!/usr/bin/env python

#template for random classifier training

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import sys
from imageprocessing import ImageAlgorithm
from hsvhistogram import HSVHistogram, BinaryHistogram

def testVideo():
    imAI = ImageAlgorithm()
    video_path = '/home/ka/Desktop/new.avi'
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):                
        # obtain video frame, display the hsv image
        ret, imAI.frame = cap.read()
        if imAI.frame != None:
            imAI.colorBound()
            cv2.imshow("output", imAI.frame)
    
            if cv2.waitKey(1) & 0XFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()

# sklearn RandomForest Classifier training results
def trainModel_sk():
    imAI = ImageAlgorithm()

    # grab the image and mask paths
    imagePaths = glob.glob('/home/ka/Desktop/images/*.jpg')
    
    # initialize the list of data and class label targets
    data = []
    target = []
    
    # initialize the image descriptor
    #desc = RGBHistogram([8, 8, 8])
    desc = BinaryHistogram([256])
    
    # loop over the image and mask paths
    for imagePath in imagePaths:
        # load the image and mask
        image = cv2.imread(imagePath)
        imAI.frame = image
        imAI.colorBound()
        image = imAI.bgr_output
    
        # describe the image
        features= desc.describe(image)
    
        # update the list of data and targets
        data.append(features)
        target.append(imagePath.split("_")[-2])
        
    np.save('dataf', data)
    np.save('targetf', target)
    
    # grab the unique target names and encode the labels
    targetNames = np.unique(target)
    le = LabelEncoder()
    target = le.fit_transform(target) # array
    #np.save("target", target)
    
    # construct the training and testing splits
    (trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
        test_size = 0.3, random_state = 42)
    
    # train the classifier
    model = RandomForestClassifier(n_estimators = 25, random_state = 84)
    model.fit(trainData, trainTarget) # both array
    #print(type(model), type(trainData), type(trainTarget))
    #print(trainData, trainTarget)
    
    #np.save('trainData', trainData)
    #np.save('trainTarget', trainTarget)
    
    # evaluate the classifier
    print (classification_report(testTarget, model.predict(testData),
        target_names = targetNames))
    print("sk model trained")
    
    testPaths = sorted(glob.glob('/home/ka/Desktop/image2/*.jpg'))
    # loop over a sample of the images
    for i in np.random.choice(np.arange(0, len(testPaths)), 15):

        # grab the image and mask paths
        testPath = testPaths[i]
    
        # load the image and mask
        image = cv2.imread(testPath)
        #mask = cv2.imread(maskPath)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
        # describe the image
        features = desc.describe(image)
    
        # predict what type of flower the image is
        res = le.inverse_transform(model.predict(features))[0]
        if "PATH" in res.upper():
            print ("I think this is a path. ")
        else:
            print("This is not a path.")
        cv2.imshow("image", image)
        cv2.waitKey(0)

# ========================= End of sk model ===================================

def main():
    trainModel_sk()
    

if __name__ == "__main__":
    main()