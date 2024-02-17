from ultralytics import YOLO
import cv2

print("OpenCV version:", cv2.__version__)
# load yolov8 model
model = YOLO('best.pt')



# load video
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)


# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

ret = True
# read frames

i = 0
width = None
height = None

while ret:
    ret, frame = cap.read()

    if (i==0): 
        height,width,layers=frame.shape
        video=cv2.VideoWriter('output.avi',-1,1,(width,height))
        i = i + 1


    if ret:
        results = model.track(frame, persist=True)
        frame_ = results[0].plot()
        video.write(frame_)

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
