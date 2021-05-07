from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import pytesseract
import cv2
import re
import os
from PIL import Image
#py my_text.py --east frozen_east_text_detection.pb
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "D:\tesseract\tessdata"'
def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	for y in range(0, numRows):
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			if scoresData[x] < args["min_confidence"]:
				continue
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	
	return (rects, confidences)


ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, default='frozen_east_text_detection.pb',
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())



(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])
# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	#vs = VideoStream(0).start()
	vs=cv2.VideoCapture(0)
	#vs= cv2.VideoCapture("http://192.168.0.102:8080/video")
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	
	vs = cv2.VideoCapture(args["video"])
fps = FPS().start()

def gray(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# blur
def blur(img) :
    img_blur = cv2.GaussianBlur(img,(5,5),0)
      
    return img_blur

def threshold(img):
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
    
    return img
def yolo_calling():
	os.system("py detect.py --source 0 --weights yolov5x.pt --conf 0.4 ")
counter=0
while True:
	ret,frame = vs.read()
	#ret, frame = vs.read()
	#frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()
	(origH, origW) = frame.shape[:2]
	
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)
	
	frame = cv2.resize(frame, (newW, newH))
	(H, W) = frame.shape[:2]
	im_gray = gray(frame)
	im_blur = blur(im_gray)
	frames = threshold(im_blur)
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	
	(rects, confidences) = decode_predictions(scores, geometry)
	
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	if len(boxes)>3:
		counter=0 #modules switching
		results=[]
		"""cv2.imwrite('a.png',frame)
		img=Image.fromarray(frame)
		img.load()"""
		for (startX, startY, endX, endY) in boxes:
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			dX = int((endX - startX) * args["padding"])
			dY = int((endY - startY) * args["padding"])
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(origW, endX + (dX * 2))
			endY = min(origH, endY + (dY * 2))
			# extract the actual padded ROI
			roi = orig[startY:endY, startX:endX]
			config = ("-l eng --oem 1 --psm 3")
			text = pytesseract.image_to_string(roi, config=config)
			#text=pytesseract.image_to_string(roi, lang='eng', config=tessdata_dir_config)
			results.append(text)
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		#results=pytesseract.image_to_string(img, lang='eng', config=tessdata_dir_config)
		results=[re.sub(r"[^A-Za-z0-9]+", "",x) for x in results]
		
		text = " ".join(results).strip()
		#print(results)
		#time.sleep(5)
		print(text)
		"""if(len(results)>10):
				os.system("py textsum.py text")
			else:
				pass"""

	else:
		print('No text found ; Proceeding with YOLO')
		counter+=1
		if(counter>40):
			vs.release()
			cv2.destroyAllWindows()	
			
			break
			
# loop over the results
	"""for ((startX, startY, endX, endY), text) in results:
		# display the text OCR'd by Tesseract
		print("OCR TEXT")
		print("========")
		print("{}\n".format(text))"""
		

    	# update the FPS counter
	fps.update()
	# show the output frame
	cv2.imshow("Text Detection", orig)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break


fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
"""if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()"""
# close all windows
choice=int(input("No text found in scenario Enter your choice? \n 1) Open OCR \n 2) Continue with Object Detection \n"))
if(choice==1):
	os.system("py my_text.py --east frozen_east_text_detection.pb")
import time
if(counter):
	#time.sleep(6)
	yolo_calling()