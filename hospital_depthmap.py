import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
# from pylibfreenect2 import Freenect2, SyncMultiFrameListener
# from pylibfreenect2 import FrameType, Registration, Frame
# from pylibfreenect2 import createConsoleLogger, setGlobalLogger
# from pylibfreenect2 import LoggerLevel


#obtain the format of the data
colorVideoCapture = cv2.VideoCapture('color.avi')
depthVideoCapture = cv2.VideoCapture('depth.avi')

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
# video = cv2.VideoWriter('color_test.avi',fourcc,20.0,(1920,1080))
numpy_hist = plt.figure()
while colorVideoCapture.isOpened():
    ####################Get the center of the hand########################
    ret, frame = colorVideoCapture.read()
    width = 1920
    height = 1080
    #frame = cv2.resize(color, (int(1920 / 3), int(1080 / 3)))
    # cv2.imshow('Color', frame) #show the frame
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
    #cv2.imshow('binary', thresh) #show the frame
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
        # print cv2.contourArea(unified_cont[i])
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

    #Find convex hull
    hull = cv2.convexHull(cnts)
    
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)
    
    #Print bounding rectangle
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    img_new = cv2.resize(img, (512, 424))

    cv2.imshow('Result', img_new)

    #print str(x)+','+str(y)+','+str(w)+','+str(h)
    ret2, depth_frame = depthVideoCapture.read()
    
    # print depth_frame.shape
    imgcopy = depth_frame.copy()
    imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)
    print imgcopy.shape
    cv2.imshow('depth', imgcopy)
    # print imgcopy.min()
    # print imgcopy.max()
    # # display hack to hide nd depth
    # msk = np.logical_and(4500 > imgcopy, imgcopy > 0)
    # msk2 = np.logical_or(imgcopy == 0, imgcopy == 4500)
    # min = imgcopy[msk].min()
    # max = imgcopy[msk].max()
    # imgcopy = (imgcopy - min) / (max - min) * 255.
    # imgcopy[msk2] = 255.
    # imgcopy = imgcopy.astype('uint8')
    # imgcopy = cv2.cvtColor(imgcopy, cv2.COLOR_GRAY2BGR)
    # # imgcopy = cv2.resize(imgcopy, (int(1920 / 3), int(1082 / 3)))
    # cv2.imshow('Depth', depth_frame)
    depth_hand = depth_frame[x:x+w, y:y+h,:]
    depth_img = cv2.rectangle(depth_frame,(x,y),(x+w,y+h),(0,255,0),2)
    # cv2.imshow('Depth_hand', depth_hand)
    ax = numpy_hist.add_subplot(1,1,1)
    ax.hist(depth_hand, bins=255)
    
    #Print execution time
    #print time.time()-start_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

colorVideoCapture.release()
depthVideoCapture.release()
# video.release()
cv2.destroyAllWindows()
