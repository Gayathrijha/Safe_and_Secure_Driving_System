import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import requests
import numpy as np
from EAR_calculator import *
from imutils import face_utils 
from imutils.video import VideoStream 
import imutils 
import dlib
import time 
import argparse 
import cv2 
import pandas as pd
import csv
from playsound import playsound
from scipy.spatial import distance as dist
import os 
from datetime import datetime

# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# All eye and mouth aspect ratio with time
ear_list=[]
total_ear=[]
mar_list=[]
total_mar=[]
ts=[]
total_ts=[]

url = "http://26.78.18.191:8080/shot.jpg"

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regarded as a blink 
EAR_THRESHOLD = 0.3
# Declare another constant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 15 
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 14

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Now, initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO] Loading the predictor...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landmarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO] Loading Camera...")
time.sleep(2) 

assure_path_exists("dataset_phonecam/")
count_sleep = 0
count_yawn = 0 

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []

while True: 

    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype = np.uint8)
    frame = cv2.imdecode(img_arr, -1)
    frame = imutils.resize(frame, width = 875)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor 
    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array 
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face 
        (x, y, w, h) = face_utils.rect_to_bb(rect) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a circle for the face 
        cv2.circle(frame, (int((x + x + w) / 2), int((y + y + h) / 2)), 3, (0, 0, 255), -1)

        # Now let's try to draw lines on the face 
        cv2.line(frame, (shape[36][0], shape[36][1]), (shape[39][0], shape[39][1]), (0, 255, 0), 1)
        cv2.line(frame, (shape[42][0], shape[42][1]), (shape[45][0], shape[45][1]), (0, 255, 0), 1)
        cv2.line(frame, (shape[48][0], shape[48][1]), (shape[54][0], shape[54][1]), (0, 255, 0), 1)

        # Calculate the EAR for both the eyes
        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend]
        mar = shape[mstart:mend]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthEAR = mouth_aspect_ratio(mar)
        
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouthEAR
        
        # Append EAR and MAR to the respective lists
        ear_list.append(ear)
        total_ear.append(ear)
        mar_list.append(mar)
        total_mar.append(mar)
        ts.append(dt.datetime.now())
        total_ts.append(dt.datetime.now())
        
        # Compute the convex hull for the left and right eye, then visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mar)

        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        # Check if EAR is below the threshold, if so, increment the blink frame counter
        if ear < EAR_THRESHOLD:
            FRAME_COUNT += 1
        else:
            # If the eyes were closed for a sufficient number of frames, increment the blink count
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                BLINK_COUNT += 1
            # Reset the eye frame counter
            FRAME_COUNT = 0

        # Check if MAR is above the threshold, if so, increment the yawn frame counter
        if mar > MAR_THRESHOLD:
            FRAME_COUNT += 1
        else:
            # If the mouth was open for a sufficient number of frames, increment the yawn count
            if FRAME_COUNT >= CONSECUTIVE_FRAMES:
                YAWN_COUNT += 1
            # Reset the yawn frame counter
            FRAME_COUNT = 0

        # Display the current blink count and yawn count on the frame
        cv2.putText(frame, "Blinks: {}".format(BLINK_COUNT), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Yawns: {}".format(YAWN_COUNT), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Blink Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

# Plotting the EAR and MAR values over time
plt.plot(ts, ear_list, label='EAR')
plt.plot(ts, mar_list, label='MAR')
plt.xlabel('Time')
plt.ylabel('EAR / MAR')
plt.title('EAR and MAR over Time')
plt.legend()
plt.show()
