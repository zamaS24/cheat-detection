import numpy as np
import time
import cv2
import os

labelsPath = 'labels.names'

LABELS = open(labelsPath).read().strip().split("\n")

COLORS = np.array([[143,0,255],[ 220, 180,  61]])

net = cv2.dnn.readNetFromDarknet('cheatingModel.cfg', 'cheatingModel.weights')
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

frame = cv2.imread('bro.jpg')

(H, W) = frame.shape[:2]

gui_confidence = .4     # initial settings
gui_threshold = .3

blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# initialize our lists of detected bounding boxes, confidences,
# and class IDs, respectively
boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
  # loop over each of the detections
  for detection in output:
    # extract the class ID and confidence (i.e., probability)
    # of the current object detection
    scores = detection[5:]
    classID = np.argmax(scores)
    confidence = scores[classID]

    # filter out weak predictions by ensuring the detected
    # probability is greater than the minimum probability
    if confidence > gui_confidence:
      box = detection[0:4] * np.array([W, H, W, H])
      (centerX, centerY, width, height) = box.astype("int")
      # use the center (x, y)-coordinates to derive the top
      # and and left corner of the bounding box
      x = int(centerX - (width / 2))
      y = int(centerY - (height / 2))

      # update our list of bounding box coordinates,
      # confidences, and class IDs
      boxes.append([x, y, int(width), int(height)])
      confidences.append(float(confidence))
      classIDs.append(classID)

# apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)
# ensure at least one detection exists
if len(idxs) > 0:
  # loop over the indexes we are keeping
  for i in idxs.flatten():
    # extract the bounding box coordinates
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    # draw a bounding box rectangle and label on the frame
    color = [int(c) for c in COLORS[classIDs[i]]]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

cv2.imshow('window',frame)
cv2.waitKey()
cv2.destroyAllWindows()