import cv2
from warnings import simplefilter

simplefilter("ignore")

TrainedFaceData =  cv2.CascadeClassifier("haarcascades_frontalfaces.xml")

#This is For a Static Single Image Detection
img = cv2.imread("assets/sample.png")
GrayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coor = TrainedFaceData.detectMultiScale(GrayScaleImg)

for (x, y, w, h) in face_coor:
	cv2.rectangle(img, (x, y), (x+w,  y+h), (0, 0, 255), 5)

cv2.imshow("Face Detected", img)
# Press any Key to Proceed the Execution
cv2.waitKey()


# This is for a Video esp. from a WebCam
webcam = cv2.VideoCapture(0)

while True:
	SucessfulFrameRead, frame = webcam.read()	

	GrayScaleImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	face_coor = TrainedFaceData.detectMultiScale(GrayScaleImg)

	for (x, y, w, h) in face_coor:
		cv2.rectangle(frame, (x, y), (x+w,  y+h), (0, 0, 255), 2)

	cv2.imshow("Face Detected", frame)
	key = cv2.waitKey(1)

	# Press 'Q' or 'q' to Quit
	if key in (81, 113):
		break

webcam.release()
