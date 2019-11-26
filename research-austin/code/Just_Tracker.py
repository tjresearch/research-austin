import cv2
import math
import numpy as np
print("Hello world")
#ncaa_2019_semifinals_maryland_vs_virginia.mp4
#game_vid.mp4
#bbox for ncaa frame 6300 pass: (235, 240, 10, 10) works
#bbox for ncaa frame 7108 pass: (510, 322, 22, 22) works until caught in front of white player
#bbox for game_vid frame 492 pass: (890, 96, 20, 10) doesnt work because it goes into the red track
#bbox for ncaa_frame 6456 pass: (275, 185, 10, 20) works with background subtractor; regular pass that is fast and is not obscured or blends in with background
#6456 pass gets stuck between 88 and 89 without background subtractor
vid_capture = cv2.VideoCapture("/C:/Users/austin/Desktop/ncaa_2019_semifinals_maryland_vs_virginia.mp4")
#game starts at frame 5700 for semifinal
#good starting frame for short pass is 6300 for semifinal
#good starting frame for normal pass is 7100 for semifinal
start_frame = 6456
end_frame = 6492
i = start_frame
tracker = cv2.TrackerCSRT_create()
vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
_, frame = vid_capture.read() #getting the first frame
cv2.imshow("frame", frame)
cv2.waitKey(0)

bbox = (275, 185, 10, 20) #initial bounding box
trackerType = "CSRT"
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0))
cv2.imshow("frame", frame)
cv2.waitKey(0)

# Process video and track objects
while vid_capture.isOpened():
  success, frame = vid_capture.read()

  if not success:
    break
  # get updated location of objects in subsequent frames
  success, bbox = tracker.update(frame)
  #draw tracked objects
  p1 = (int(bbox[0]), int(bbox[1]))
  p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
  cv2.rectangle(frame, p1, p2, (255,0,0))
  # show frame
  cv2.imshow('CSRTTracker', frame)
  print("i")
  print(i)
  if cv2.waitKey(0) & 0xFF == 27:  # Esc pressed
    break
  if i > end_frame:
    break
  i+=1
vid_capture.release()
cv2.destroyAllWindows()
