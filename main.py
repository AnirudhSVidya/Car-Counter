import cv2
import numpy as np


#Capturing Video
vid=cv2.VideoCapture('video.mp4')

minwidthrect=80
minheightrect=80

#Line
countlineposition=550
#Initialising substructor
algo = cv2.createBackgroundSubtractorMOG2() #removes the background of the vehicles   
while True:
    ret, frame1 = vid.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #converts video to gray
    blur=cv2.GaussianBlur(gray,(9,9),14)  #makes the video smoother by reducing sharpness hence decreasing clarity. NOTE: more kernal value means algo will pick up less things(only odd values)
    #Now applying on frames
    Sub_img=algo.apply(blur) #using the algorithm on blur
    dilated_img=cv2.dilate(Sub_img,np.ones((1,1)))  #grows the img by increeasing size of neighobouring pixels
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)) #usage of cv2.morph_ellipse tells kernal to dilate in ellipse shape
    dilate=cv2.morphologyEx(dilated_img,cv2.MORPH_CLOSE,kernel) #performs morpological closing operation on kernal. CLOSING MEANS TO DILATE AND ERODE WITH SAME STRUCTURING ELEMENT(KERNEL)
    dilate=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel)
    countershape,h=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #findcontours is used to create boundaries in a binary image. cv2.RETR_TREE is a retrival method that gives output in heirarchal manner, CHAIN_APPROX_SIMPLE reduces no of bar points needed to represent the image making it less detailed
    cv2.line(frame1,(25,countlineposition),(12000,countlineposition),(200,0,0),2) #Creating Line in image
    for (i,c) in enumerate(countershape):
        (x,y,w,h)=cv2.boundingRect(c) #creates a tuple contianing details of rectangle to show
        Checkcounter = (w>=minwidthrect) and (h>=minheightrect) # chcks if rectangle shld be made(find cars)
        if not Checkcounter:
            continue
        if w > 5.5* minwidthrect:
        # Try splitting the contour horizontally
            mid_x = x + w // 2
            cv2.rectangle(frame1, (x, y), (mid_x, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame1, (mid_x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,200),2)
    #cv2.imshow('Detector',dilate)
    cv2.imshow('Original',frame1)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()
vid.release() 
