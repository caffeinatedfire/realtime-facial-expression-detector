# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 02:31:14 2019

@author: Caffeinated Fire
"""

import cv2
import os

currentDirectory = "D:/Facial Expression Detector/training_images/sad/"
                    
images = os.listdir(currentDirectory)
i = 1
    
for filename in os.listdir(currentDirectory): 
    dst = str(i) + ".jpg"
    src = currentDirectory + filename 
    dst = currentDirectory + dst 
    
    os.rename(src, dst) 
    i += 1
    
cv2.destroyAllWindows()
