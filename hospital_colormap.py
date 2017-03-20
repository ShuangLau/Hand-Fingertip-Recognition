import cv2
import numpy as np
import time
import sys
# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# from pylibfreenect2 import createConsoleLogger, setGlobalLogger
# from pylibfreenect2 import LoggerLevel


#obtain the format of the data
videoCapture = cv2.VideoCapture('color.avi')

def find_if_close(cnt1,cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 40 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

def nothing(x):
    pass

# Function to find angle between two vectors
def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

# Function to find distance between two points in a list of lists
def FindDistance(A,B): 
 return np.sqrt(np.power((A[0]-B[0]),2) + np.power((A[1]-B[1]),2)) 

def getKey(item):
    return item[1]

def angle_rad(v1, v2):
    return np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

def deg2rad(angle_deg):
    return angle_deg/180.0*np.pi

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')

# Starting with 100's to prevent error while masking
h,s,v = 100,100,100

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,nothing)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,nothing)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,nothing)
 
# #get the fps and the size of the frame
# fps = videoCapture.get(cv2.CV_CAP_PROP_FPS)
# size = (int(videoCapture.get(cv2.CV_CAP_PROP_FRAME_WIDTH)), 
#         int(videoCapture.get(cv2.CV_CAP_PROP_FRAME_HEIGHT)))

#designate the format of the video, I420-avi, MJPG-mp4
#videoWriter = cv2.VideoWriter('oto_other.mp4', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)
 
#read the frame
# success, frame = videoCapture.read()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('color_point_signal.avi',fourcc,20.0,(1920,1080))
while videoCapture.isOpened():
    ret, frame = videoCapture.read()
    width = 1920
    height = 1080
    #frame = cv2.resize(color, (int(1920 / 3), int(1080 / 3)))
    cv2.imshow('Color', frame) #show the frame
    #videoWriter.write(frame) #write the frame
    #Blur the image
    blur = cv2.blur(frame,(3,3))
 	
 	#Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    
    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([9,50,50]),np.array([15,255,255]))

    
    #Kernel matrices for morphological transformation   
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #Perform morphological transformations to filter out the background noise
    #Dilation increase skin color area
    #Erosion increase skin color area
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)
    cv2.imshow('binary', thresh) #show the frame
    #Find contours of the filtered frame
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    
    
    
	#Find Max contour area (Assume that hand is in the frame)
    #max_area=100
    # max_area = -1
    # ci=0	
    # for i in range(len(contours)):
    #     cnt=contours[i]
    #     area = cv2.contourArea(cnt)
    #     if(area>max_area):
    #         max_area=area
    #         ci=i 

    #Preprocess the contours, merge the contours which are too close into the same one
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))
    #print LENGTH
    if LENGTH == 0:#When there is no contours
        continue

    for i, cnt1 in enumerate(contours):
        x = i
        if i!=LENGTH-1:#if the contour is not the last one
            for j, cnt2 in enumerate(contours[i+1:]):#check the contours behind it
                x=x+1#the index of the cnt2
                ctr_dist = find_if_close(cnt1, cnt2)
                if ctr_dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:#two contours are not close
                    if status[x]==status[i]:
                        status[x] = i+1

    unified = []
    unified_cont = []
    maximum = int(status.max())+1
    for i in xrange(maximum):
        pos = np.where(status==i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            unified_cont.append(cont)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    # cv2.drawContours(frame,unified,-1,(0,255,0),2)
    # cv2.drawContours(thresh,unified,-1,255,-1)
    # cv2.imshow('Dilation',frame)
    # cv2.imshow('Thresh',thresh)
    #Find Min contour area(Assume that hand is in the frame)
    min_area = 10000000
    ci=0    
    for i in range(len(unified_cont)):
        print cv2.contourArea(unified_cont[i])
        #If contours are too small or large, ignore them:
        if cv2.contourArea(unified_cont[i])<1000:
            continue
        elif cv2.contourArea(unified_cont[i])>8000:
            continue
        #print cv2.contourArea(unified_cont[i])
        cnt=unified_cont[i]
        area = cv2.contourArea(cnt)
        if(area<min_area):
            min_area=area
            ci=i 
            
	#Smallest area contour  
    cnts = unified_cont[ci]
    #print cv2.contourArea(cnts)
    #print cnts.size()
    #Draw Contours
    #cv2.drawContours(frame, cnts, -1, (122,122,0), 3)
    #cv2.imshow('Dilation',frame)


    #Orientation is the angle at which object is directed.
    #Following method also gives the Major Axis and Minor Axis lengths.
    (x_1,y_1),(MA,ma),angle = cv2.fitEllipse(cnts)
    print '(x_1,y_1),(MA,ma),angle:'
    print repr(x_1), repr(y_1), repr(MA), repr(ma), angle

    font = cv2.FONT_HERSHEY_SIMPLEX
    if angle < 100. and angle > 80.:
        cv2.putText(frame, 'Pointing!', (50, 50), font, 2, (0, 125, 255), 2)

    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #print str(x)+','+str(y)+','+str(w)+','+str(h)

    #Get defect points and draw them in the original image
    thresh_deg = 90.0
    num_fingers = -2
    FarDefect = []

    if defects is None:
        num_fingers = 0
    elif len(defects) <= 2:
        num_fingers = 0
    else:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnts[s][0])
            #cv2.circle(frame, start, 5, [100,100,255], 1)#yellow
            end = tuple(cnts[e][0])
            #cv2.circle(frame, end, 5, [255,100,100], 1)#yellow
            far = tuple(cnts[f][0])
            FarDefect.append(far)
            cv2.line(frame,start,end,[0,255,0],1)
            #cv2.circle(frame,far,3,[100,255,255],-1)
        #sorted(FarDefect, key=getKey, reverse=True)
        #FarDefect = FarDefect[0:4]
        #print FarDefect[0]
        #print FarDefect[1]
        #print FarDefect[2]
        #print FarDefect[3]  '''or max(FindDistance(start, far), FindDistance(end, far)) < 0.4*h'''

            # if angle is below a threshold, defect point belongs
            # to two extended fingers
            if angle_rad(np.subtract(start, far), np.subtract(end, far)) < deg2rad(thresh_deg) or max(FindDistance(start, far), FindDistance(end, far)) >= 0.4*min(w, h):
                # increment number of fingers
                #num_fingers = num_fingers + 1
                #draw point as green
                num_fingers = num_fingers + 1
                cv2.circle(frame, far, 5, [100,255,255], -1)#yellow
            else:
                # draw point as red
                cv2.circle(frame, far, 5, [255,0,0], -1)#red
    ##### Show final image ########
    #cv2.drawContours(frame, cnts, -1, (122,122,0), 3)
    cv2.imshow('Dilation',frame)
    video.write(frame)
 #    num_fingers = min(5, num_fingers)

 #    #Print number of pointed fingers
 #    font = cv2.FONT_HERSHEY_SIMPLEX
 #    cv2.putText(frame,str(num_fingers),(50,50),font, 2, (255,255,0), 2)

	# #Find moments of the largest contour
 #    moments = cv2.moments(cnts)

 #    #Central mass of first order moments
 #    if moments['m00']!=0:
 #        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
 #        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
 #    centerMass=(cx,cy)    
 #    #Draw center mass
 #    cv2.circle(frame,centerMass,7,[100,0,255],2)
 #    font = cv2.FONT_HERSHEY_SIMPLEX
 #    cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)

 #    #cv2.imshow('Dilation0',frame)
    
 #    #Distance from each finger defect(finger webbing) to the center mass
 #    distanceBetweenDefectsToCenter = []
 #    for i in range(0,len(FarDefect)):
 #        x = np.array(FarDefect[i])
 #        centerMass = np.array(centerMass)
 #        distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
 #        distanceBetweenDefectsToCenter.append(distance)
    
 #    #Get an average of three shortest distances from finger webbing to center mass
 #    sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
 #    AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
 
 #    #Get fingertip points from contour hull
 #    #If points are in proximity of 80 pixels, consider as a single point in the group
 #    finger = []
 #    for i in range(0,len(hull)-1):
 #        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
 #            if hull[i][0][1] <500 :
 #                finger.append(hull[i][0])
    
 #    #The fingertip points are 5 hull points with largest y coordinates  
 #    finger =  sorted(finger,key=lambda x: x[1])   
 #    fingers = finger[0:5]
    
 #    #Calculate distance of each finger tip to the center mass
 #    fingerDistance = []
 #    for i in range(0,len(fingers)):
 #        distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
 #        fingerDistance.append(distance)
    
 #    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
 #    #than the distance of average finger webbing to center mass by 130 pixels
 #    result = 0
 #    for i in range(0,len(fingers)):
 #        if fingerDistance[i] > AverageDefectDistance+130:
 #            result = result +1
    
 #    #Print number of pointed fingers
 #    #cv2.putText(frame,str(result),(100,100),font,2,(255,255,255),2)
    
 #    #show height raised fingers
 #    #cv2.putText(frame,'finger1',tuple(finger[0]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger2',tuple(finger[1]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger3',tuple(finger[2]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger4',tuple(finger[3]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger5',tuple(finger[4]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger6',tuple(finger[5]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger7',tuple(finger[6]),font,2,(255,255,255),2)
 #    #cv2.putText(frame,'finger8',tuple(finger[7]),font,2,(255,255,255),2)
        
    
    
 #    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
 #    cv2.imshow('Dilation',frame)

    ###############################
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #Print execution time
    #print time.time()-start_time
    # cv2.waitKey(1000/int(fps)) #delay
    # success, frame = videoCapture.read() #capture the next frame
videoCapture.release()
video.release()
cv2.destroyAllWindows()