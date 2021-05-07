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
#py my_text.py --east frozen_east_text_detection.pb
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "D:\tesseract\tessdata"'
def decode_predictions(scores, geometry):
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)


ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
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

# threshold
def threshold(img):
    #pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
    
    return img
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	ret,frame = vs.read()
	#ret, frame = vs.read()
	#frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()
	(origH, origW) = frame.shape[:2]
	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)
	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))
	(H, W) = frame.shape[:2]
    # construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	im_gray = gray(frame)
	im_blur = blur(im_gray)
	frames = threshold(im_blur)
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	
	(rects, confidences) = decode_predictions(scores, geometry)
	
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	if len(boxes)>3:
		counter=0 #modules switching
		results=[]
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
		results=[re.sub(r"[^A-Za-z0-9]+", "",x) for x in results]
		text = " ".join(results).strip()

		print(text)
		

	
			
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


# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# if we are using a webcam, release the pointer
"""if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()"""
# close all windows
