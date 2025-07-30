import cv2
import numpy as np


#Capturing Video
vid=cv2.VideoCapture('video.mp4')

minwidthrect=50
minheightrect=57

#Line
countlineposition=460
#Initialising substructor
algo = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=24,detectShadows=False)#removes the background of the vehicles 

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detect=[] #list to store the centers of the cars
offset=9 #allowable pixel error
counter=0
counted_center=[]
dots_to_show = []
while True:
    ret, frame1 = vid.read()
    if not ret:
        break
    gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #converts video to gray
    blur=cv2.GaussianBlur(gray,(3,3),6)  #makes the video smoother by reducing sharpness hence decreasing clarity. NOTE: more kernal value means algo will pick up less things(only odd values)
    #Now applying on frames
    Sub_img=algo.apply(blur) #using the algorithm on blur
    dilated_img=cv2.dilate(Sub_img,np.ones((5,5)))  #grows the img by increeasing size of neighobouring pixels
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)) #usage of cv2.morph_ellipse tells kernal to dilate in ellipse shape
    dilateq=cv2.morphologyEx(dilated_img,cv2.MORPH_OPEN,kernel) #performs morpological closing operation on kernal. CLOSING MEANS TO DILATE AND ERODE WITH SAME STRUCTURING ELEMENT(KERNEL)
    dilate=cv2.morphologyEx(dilateq,cv2.MORPH_CLOSE,kernel)
    contourshape,h=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #findcontours is used to create boundaries in a binary image. cv2.RETR_TREE is a retrival method that gives output in heirarchal manner, CHAIN_APPROX_SIMPLE reduces no of bar points needed to represent the image making it less detailed
    cv2.line(frame1,(25,countlineposition),(12000,countlineposition),(200,0,0),3) #Creating Line in image
    for (i, c) in enumerate(contourshape):
        (x, y, w, h) = cv2.boundingRect(c)
        center = center_handle(x, y, w, h)
        if w >= minwidthrect and h >= minheightrect:
            aspect_ratio = float(w) / h

           
            if w > 4.3 * minwidthrect:
                for i in range(2):
                    mid_x = x + i*(w // 2)
                    cv2.rectangle(frame1, (x, y), (mid_x, y + h), (0, 255, 0), 2)
                    cv2.rectangle(frame1, (mid_x, y), (x + w, y + h), (0, 255, 0), 2)
                    detect.append(center)
            else:
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 200), 2)

            
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)
    
    for center in detect[:]:
        x, y = center
        if (countlineposition - offset) < y < (countlineposition + offset):
            if center not in counted_center:
                counted_center.append(center)
                counter += 1
                detect.remove(center)
                print("Vehicle Counter:" + str(counter))
        
    cv2.putText(frame1,"Vehicles passed:"+str(counter),(470,70),cv2.FONT_ITALIC,1.4,(0,0,0),3)


    cv2.imshow('Detector',dilate)
    cv2.imshow('Original',frame1)
    if cv2.waitKey(10)==ord('q'):
        break
cv2.destroyAllWindows()
vid.release() 
