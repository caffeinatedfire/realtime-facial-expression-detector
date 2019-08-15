# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:34:49 2019

@author: Caffeinated Fire
"""

import os
import tensorflow as tf
import numpy as np
import cv2

RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def main():
    print("initiating......")

    if not checkIfNecessaryPathsAndFilesExist():
        return

    # list classifications from the labels file
    classifications = []
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        classification = currentLine.rstrip()
        classifications.append(classification)

    print("classifications = " + str(classifications))

    # load graph
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph
        _ = tf.import_graph_def(graphDef, name='')
    
    cap = cv2.VideoCapture(0)

    with tf.Session() as sess:
        while True:
            ret, im = cap.read()
            faces = classifier.detectMultiScale(im)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,255), 2)
                roi_face = im[y:y+h, x:x+w]
                
                roi_face = cv2.resize(roi_face, (300, 300), interpolation = cv2.INTER_AREA)
                roi_face = cv2.cvtColor(roi_face, cv2.COLOR_BGR2GRAY)                    
                cv2.imwrite("currentTestFile.jpg", roi_face)
            
            openCVImage = cv2.imread("currentTestFile.jpg")
            
            # get final tensor from graph
            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # convert OpenCV image (numpy array) to TensorFlow image
            tfImage = np.array(openCVImage)[:, :, 0:3]
            
            # run the network to get the predictions
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            #adding weights to adjust for training data, ethnicity and lighting
            #add more array indices if data set was created with more expressions
            #array indice corresponds to classifications = []
            predictions[0][0] += 0.25
            predictions[0][1] -= 0.00
            predictions[0][2] -= 0.15
            
            # sort predictions from most confidence to least confidence
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            MostLikelyPrediction = sortedPredictions[0]
            strClassification = classifications[MostLikelyPrediction]
            confidence = 100*predictions[0][MostLikelyPrediction]
            writeResult(im, strClassification + ", " + "{0:.2f}".format(confidence) + "%")
            
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]
                confidence = predictions[0][prediction] 
                print(strClassification + ", " + "{0:.2f}".format(confidence))
            
            print('')

            cv2.imshow("Test", im)

# =============================================================================         
#             PRESS ESC ON "TEST" WINDOW TO EXIT
# =============================================================================
            key = cv2.waitKey(10)
            if key == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()
    print("updating tfevent files, please wait......")
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()
    print("done")

# =============================================================================
def checkIfNecessaryPathsAndFilesExist():

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False

    print('All Paths and Files Exist')
    return True

# =============================================================================
def writeResult(openCVImage, resultText):
    
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontThickness = 2
    
    cv2.putText(openCVImage, resultText, (180, 450), fontFace, fontScale, (255,0,0), fontThickness)
    cv2.putText(openCVImage, 'Caffeinated Fire', (20,50), fontFace, fontScale*1.5, (0,0,0), fontThickness*2)
    
# =============================================================================
if __name__ == "__main__":
    main()
