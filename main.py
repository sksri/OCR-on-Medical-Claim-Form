# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 14:38:00 2021

@author: S7012205
"""
# Importing libraries
import cv2
import numpy as np
import pytesseract
import os
from openpyxl import load_workbook

template = 'blank_form.jpg'
per = 100
pixelThreshold=40

roi = [[(54, 262), (462, 290), 'text', 'Patient Name'], 
       [(468, 266), (518, 290), 'text', 'Patient DOB(MM)'],
       [(524, 266), (562, 290), 'text', 'Patient DOB(DD)'],
       [(568, 266), (620, 290), 'text', 'Patient DOB(YY)'],
       [(638, 272), (672, 288), 'box', 'Patient Gender(M)'], 
       [(708, 266), (740, 288), 'box', 'Patient Gender(F)'], 
       [(52, 312), (460, 340), 'text', 'Patient Street'], 
       [(52, 358), (404, 382), 'text', 'Patient City'], 
       [(408, 360), (464, 384), 'text', 'Patient State'], 
       [(52, 400), (232, 434), 'text', 'Patient Zipcode'], 
       [(238, 402), (462, 436), 'text', 'Patient Telephone'], 
       [(756, 214), (1028, 242), 'text', 'InsuranceID'], 
       [(52, 644), (460, 674), 'text', 'InsurancePlan'], 
       [(758, 982), (1182, 1008), 'text', 'PriorAuthorizationNumber'], 
       [(1013, 1082), (1185, 1100), 'text', 'NPI']]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\s7012205\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

# Loading blank claim form image
img_blank = cv2.imread(template)
w,h,c = img_blank.shape
#img_blank = cv2.resize(img_blank,(w//2, h//2))

# Creating detector
orb = cv2.ORB_create(3000)
# Found out key points and descriptor
kp1, des1 = orb.detectAndCompute(img_blank, None)

#impKp1 = cv2.drawKeypoints(img_blank, kp1, None)
#cv2.imshow('KeyPointsQuery', impKp1)

# Loading filled claimed form images
path = 'UserForms'
output = 'Output'
myPicList = os.listdir(path)
print(myPicList)

for j,y in enumerate(myPicList):
    img_filled = cv2.imread(path + '/' + y)
    #img_filled = cv2.resize(img_filled,(w//2, h//2))
    #cv2.imshow(y, img_filled)
    
    # Creating keypoints and descriptors
    kp2, des2 = orb.detectAndCompute(img_filled, None)
    
    # Creating bruitforce matcher to actually match the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key= lambda x: x.distance)
    
    # Taking 25% matcher and draw the points
    good = matches[:int(len(matches)*(per/100))]
    img_match = cv2.drawMatches(img_filled, kp2, img_blank, kp1, good[:], None, flags=2)
    #img_match = cv2.resize(img_match,(w//2, h//2))
    #cv2.imshow(y, img_match)
    
    # Find the source and destination points
    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # Find the matrix to match the source and destination points
    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    # Find the birds eye view of the forms
    img_scan = cv2.warpPerspective(img_filled,M,(w,h))
    #img_scan = cv2.resize(img_scan,(w//2, h//2))
    #cv2.imshow(y, img_scan)
    
    img_show = img_scan.copy()
    img_mask = np.zeros_like(img_show)
     
    # to capture text data for images
    myData = []
    
    print(f'############### Extracting data from claim form {j+1} ###############')
    
    for x,r in enumerate(roi):
        
        cv2.rectangle(img_mask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        img_show = cv2.addWeighted(img_show,0.99,img_mask,0.1,0)
        
        img_crop = img_scan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        #cv2.imshow(str(x), img_crop)
        
        if r[2] == 'text':
            print(f'{r[3]} : {pytesseract.image_to_string(img_crop)}')
            myData.append(pytesseract.image_to_string(img_crop))
        if r[2] == 'box':
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            img_thresh = cv2.threshold(img_gray,170,255,cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(img_thresh)
            #print(totalPixels)
            if totalPixels>pixelThreshold:
                totalPixels = str(1)
            else:
                totalPixels= str(0)
            print(f'{r[3]} : {totalPixels}')
            myData.append(totalPixels)
        cv2.putText(img_show,str(myData[x]),(r[0][0],r[0][1]),cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)
    
    # Saving records in excel shet        
    wb = load_workbook('MedicalClaimFormRecords.xlsx')
    ws = wb.worksheets[0]
    ws.append(myData)    
    wb.save('MedicalClaimFormRecords.xlsx')
    print()
    print(f'{j+1} record stored successfully in a file.')

    
    #img_show = cv2.resize(img_show,(w//2, h//2))
    #cv2.imshow(y+'2', img_show)   
    cv2.imwrite(output + '/' + y, img_show)

wb.close()        

#cv2.imshow('Output',img_blank)
cv2.waitKey(0)

