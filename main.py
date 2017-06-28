#!/usr/local/bin/ python
#face detector demo video
#Author: Xiaohong Zhao
#Date: 2017/6/27
import sys
import numpy as np
import cv2
import libpysunergy
from PIL import Image, ImageDraw
from PIL import ImageFont 
import commands as cm


def predict(ori, dest):
        avi = dest.split('.')[0]+'.avi'
	threshold = 0.24

	#load video
	cap = cv2.VideoCapture(ori)
	fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        print fps


	#load face detector
	net1, names = libpysunergy.load('data/face.data', 'cfg/yolo-face.cfg', 'weights/yolo-face.weights')


	#load AGE models
	net2,names2 = libpysunergy.load("data/age.data","cfg/age.cfg", "weights/age.weights")
	net3,names3 = libpysunergy.load("data/gender.data","cfg/gender.cfg", "weights/gender.weights")
	net4,names4 = libpysunergy.load("data/race.data","cfg/race.cfg", "weights/race.weights")
	top = 1



	# Define the codec and create VideoWriter object
	fourcc =  cv2.cv.CV_FOURCC('M','J','P','G')
	videoWriter = cv2.VideoWriter(avi,fourcc,fps,(width,height))

	font = ImageFont.truetype("Roboto-Regular.ttf",20)

	count = 1 
	#face detection
	while(1):
	    print count
	    count += 1
	    ret, frame = cap.read()
            if not ret:
                break
	    cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
	    pil_im = Image.fromarray(cv2_im)
	    draw = ImageDraw.Draw(pil_im)
	    

	    if ret == True:
		    (h, w, c) = frame.shape
		    dets = libpysunergy.detect(frame.data, w, h, c, threshold, net1, names)
		    #crop face and predict AGE
		    for i in range(0, len(dets)):
		        if dets[i][0] == 'face':
		            box = dets[i][2:6]
		            x0 = int(box[0])
		            x1 = int(box[1])
		            y0 = int(box[2])
		            y1 = int(box[3])
		            faceimg = frame[y0:y1, x0:x1].copy()
		            (h, w, c) = faceimg.shape
                           
		            #draw bounding box
		            draw.rectangle(((x0,y0),(x1,y1)),outline = "red", )
			    draw.rectangle(((x0+1,y0+1),(x1-1,y1-1)),outline = "red", )
		            dets2 = libpysunergy.predict(faceimg.data, w, h, c, top, net2, names2)
		            age = dets2[0][0]
		            dets3 = libpysunergy.predict(faceimg.data, w, h, c, top, net3, names3)
		            gender = dets3[0][0]
		            dets4 = libpysunergy.predict(faceimg.data, w, h, c, top, net4, names4)
		            race = dets4[0][0]
                            
		            #write classification
		            draw.text((x0,y0 - 60),'Age: ' + age,(255,0,0), font = font)
		            draw.text((x0,y0 -40),'Gender: ' + gender, (255,0,0), font = font)
		            draw.text((x0,y0 - 20),'Race: ' + race, (255,0,0), font = font)  

		    pil_im = cv2.cvtColor(np.array(pil_im),cv2.COLOR_RGB2BGR)        
		    videoWriter.write(pil_im)
            
		    
	    else:
		break
		
	cap.release()
	libpysunergy.free(net1)
	libpysunergy.free(net2)
	libpysunergy.free(net3)
	libpysunergy.free(net4)
        #convert avi video to mp4
        print('converting avi video to mp4 ...')
	cm.getstatusoutput('ffmpeg -i ' + avi +' -c:v libx264 -crf '+ str(int(fps)) +' -preset slow -c:a libfdk_aac -b:a 192k -ac 2 ' + dest)
        cm.getstatusoutput('rm ' + avi)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("usage: originalvideo resultvideo") 
	sys.exit()   
    predict(sys.argv[1], sys.argv[2])
 
