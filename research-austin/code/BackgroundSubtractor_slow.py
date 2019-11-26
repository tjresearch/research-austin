import cv2
import math
import time
import numpy as np

#####TIMES AND DATA FOR VIDEOS
#ncaa_2019_semifinals_maryland_vs_virginia.mp4 ##TITLE OF PRO LAX GAME
#starts at frame 5700
#game_vid.mp4 ##TITLE OF HIGH SCHOOL LAX GAME

#########SAMPLE PASSES
#bbox for ncaa frame 6300 pass: (235, 240, 10, 10) works
#bbox for ncaa frame 7108 pass: (510, 322, 22, 22) works until caught in front of white player
#bbox for game_vid frame 492 pass: (890, 96, 20, 10) doesnt work because it goes into the red track
#bbox for ncaa_frame 6456 pass: (275, 185, 10, 20) works with background subtractor; regular pass that is fast and is not obscured or blends in with background
#good starting frame for short pass is 6300 for semifinal
#good starting frame for normal pass is 7100 for semifinal

def averageBGR(frame1, b1, g1, r1):
    for a in range(0, frame1.shape[0],100):
        for j in range(0, frame1.shape[1],100):
           k = frame1[a,j]
           b1.append(k[0]) #because openCV reports in BGR
           g1.append(k[1])
           r1.append(k[2])
    #print("done")
    average_r = int(sum(r1)/len(r1))
    average_g = int(sum(g1)/len(g1))
    average_b = int(sum(b1)/len(b1))

    return [average_b, average_g, average_r]

def background_subtractor(testnpArray1, frame1, average_b1, average_g1, average_r1, background_thresholder1):
    for a in range(frame1.shape[0]):
        for j in range(frame1.shape[1]):
            k = frame1[a, j]
            distance = math.sqrt( (average_r1-k[0])**2 + (average_g1-k[1])**2 + (average_b1-k[2])**2 )
            #print(distance) #try threshold of 70
            if(distance >= background_thresholder1):
                testnpArray1[a][j] = (255,255,255)
            else:
                testnpArray1[a][j] = (0, 0, 0)

    return testnpArray1

def main():
    start_time = time.time()

    #DECLARE/DEFINE
    vid_capture = cv2.VideoCapture("/C:/Users/austin/Desktop/ncaa_2019_semifinals_maryland_vs_virginia.mp4")
    start_frame = 6456
    end_frame = 6492
    i = start_frame
    tracker = cv2.TrackerCSRT_create()
    r = []
    g = []
    b = []
    testnpArray = np.zeros((720, 1820, 3), np.uint8)
    background_thresholder = 90

    #BEGIN ANALYSIS
    vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame = vid_capture.read()

    [average_b, average_g, average_r] = averageBGR(frame, b, g, r)

    #APPLY BACKGROUND THRESHOLD
    testnpArray = background_subtractor(testnpArray, frame, average_b, average_g, average_r, background_thresholder)
    elapsed_time = time.time() - start_time
    print(elapsed_time)

    ## INIT TRACKING FOR 1ST FRAME
    bbox = (275, 185, 10, 20) #initial bounding box
    trackerType = "CSRT"
    tracker = cv2.TrackerCSRT_create()
    tracker.init(testnpArray, bbox)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0))
    #cv2.imshow("frame", frame)
    #cv2.waitKey(0)

    ## TRACKING FOR SUBSEQUENT FRAMES
    while vid_capture.isOpened():
      r = []
      g = []
      b = []
      success, frame = vid_capture.read()

      if not success:
        break

      [average_b, average_g, average_r] = averageBGR(frame, b, g, r)

      ## BACKGROUND SUBTRACTION
      testnpArray = background_subtractor(testnpArray, frame, average_b, average_g, average_r, background_thresholder)

      # UPDATE LOCATION OF BALL IN SUBSEQUENT FRAME
      success, bbox = tracker.update(testnpArray)

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

if __name__ == '__main__':
    main()
