# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:35:41 2019

@author: Caffeinated Fire
"""

import cv2
import os

currentDirectory = "D:/Facial Expression Detector/training_images_mix_color/angry_disgust/"
                    
images = os.listdir(currentDirectory)
i = 1
    
for filename in images:
    image = currentDirectory + filename
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(filename, img)
    
    print(str(i) + " " + filename)
    i += 1
    
cv2.destroyAllWindows()
