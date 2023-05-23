import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("http://192.168.16.114:8080/video")
while True:
    ret, frame = cap.read()
    # change into high resolution
    frame = cv2.resize(frame, (1080, 720))
    # draw a rectangle on the right side of the frame
    cv2.rectangle(frame, (720, 0), (1080, 720), (0, 255, 0), 2)
    # get the right side of the frame
    right_frame = frame[0:720, 720:1080]
    # detect the object
    results = model(right_frame)
    # get the bounding box
    frame_predicted =results[0].plot()
    # add this frame into the original frame
    frame[0:720, 720:1080] = frame_predicted
    # draw the bounding box
    # results.plot()
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break