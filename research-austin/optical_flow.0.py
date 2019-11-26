import cv2
import math
import time
import numpy as np
#import cudamat
#import gnumpy as np

#"/C:/Users/austi/PycharmProjects/LacrosseBallTracking/ncaa_2019_semifinals_maryland_vs_virginia.mp4"# filename for lacrosse video

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
def averageBGR(frame1):
    r = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 2])
    g = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 1])
    b = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 0])

    average_r = int(np.sum(r)/np.size(r))
    average_g = int(np.sum(g)/np.size(g))
    average_b = int(np.sum(b)/np.size(b))

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
    filename = input("Enter full path name to video")
    start_time = time.time()

    #DECLARE/DEFINE
    vid_capture = cv2.VideoCapture(filename)
    start_frame = 6456
    end_frame = 6492
    i = start_frame
    tracker = cv2.TrackerCSRT_create()
    r = np.array(0)
    g = np.array(0)
    b = np.array(0)
    testnpArray = np.zeros((720, 1280, 3), np.uint8)
    background_thresholder = 90
    center = (280, 195)
    dxdy = (0, 0)

    #BEGIN ANALYSIS
    vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame = vid_capture.read()

    [average_b, average_g, average_r] = averageBGR(frame)
    #print(average_r)
    #print(average_g)
    #print(average_b)

    ## APPLY BACKGROUND THRESHOLD ##
    testnpArray = background_subtractor(testnpArray, frame, average_b, average_g, average_r, background_thresholder)

    elapsed_time = time.time() - start_time
    print(elapsed_time)

    ## INIT TRACKING FOR 1ST FRAME
    bbox = (275, 185, 10, 20) #initial bounding box
    trackerType = "CSRT"
    tracker = cv2.TrackerCSRT_create()
    tracker.init(testnpArray, bbox)
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0))
    cv2.circle(frame, center, 2, (0, 255, 0))
    #cv2.imshow("frame", frame)
    #cv2.waitKey(0)


    ## TRACKING FOR SUBSEQUENT FRAMES
    while vid_capture.isOpened():
      success, frame = vid_capture.read()
      if not success:
        break
      old_bbox = bbox
      [average_b, average_g, average_r] = averageBGR(frame)

      ## BACKGROUND SUBTRACTION
      testnpArray = background_subtractor(testnpArray, frame, average_b, average_g, average_r, background_thresholder)
      if dxdy == (0,0): #base the first delta value off of background subtractor
            success, bbox = tracker.update(testnpArray)
            new_center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
            dxdy = tuple(np.subtract(new_center, center))
            center = new_center
      else: #test value of old delta plus position and goal is to weigh this greater than background subtractor
          # or if tracker gives location much more varied than previous delta, than invalidate background subtractor result
            attempted_center = tuple(np.rint(np.add(center, dxdy)).astype(int))
            attempted_bbox = (int(new_center[0]-bbox[2]/2), int(new_center[1]-bbox[3]/2), int(bbox[2]), int(bbox[3]))
            cv2.rectangle(frame, (attempted_bbox[0], attempted_bbox[1]), (attempted_bbox[0]+attempted_bbox[2], attempted_bbox[1]+attempted_bbox[3]), (0, 0, 255))
            cv2.circle(frame, attempted_center, 2, (0, 0, 255))
            cv2.imshow("frame", frame)
            cv2.waitKey(0)
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
