import cv2
import math
import time
import numpy as np
#import cudamat
#import gnumpy as np

#"/C:/Users/austi/PycharmProjects/LacrosseBallTracking/ncaa_2019_semifinals_maryland_vs_virginia.mp4"# filename for pro lacrosse video
#"/C:/Users/austi/PycharmProjects/LacrosseBallTracking/game_vid.mp4" filename for high school lacrosse video

#########TIMES AND DATA FOR VIDEOS
#ncaa_2019_semifinals_maryland_vs_virginia.mp4 ##TITLE OF PRO LAX GAME
#starts at frame 5700
#game_vid.mp4 ##TITLE OF HIGH SCHOOL LAX GAME

#########SAMPLE PASSES
#bbox for ncaa frame 6456 to 6492 pass: (275, 185, 10, 20), (280, 195) works
#bbox for ncaa frame 7108 to 7127 pass: (510, 322, 22, 22),  (521, 331) works until caught in front of white player ##works now
#bbox for ncaa frame 6300 to 6318 pass: (230, 230, 22, 22), (241, 241) works
#bbox for ncaa frame 6300
#bbox for game_vid frame 493 to 514 pass: (850, 85, 20, 10), (860, 90) doesnt work because it goes into the red track, still does not work probably because faint color

#########RESEARCH QUESTIONS
#how to determine the dimensions of the bbox
#how to treat obscurity

#########RESEARCH IDEAS
###if dxdy magnitude gets small suddenly enough, it probably means someone or something ended the pass, within the box, detect if there is a lot of non background color and from that determine if it is likely someone caught the pass

def averageBGR(frame1):
    r = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 2])
    g = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 1])
    b = np.array(frame1[0:frame1.shape[0]:100, 0:frame1.shape[1]:100, 0])

    average_r = int(np.sum(r)/np.size(r))
    average_g = int(np.sum(g)/np.size(g))
    average_b = int(np.sum(b)/np.size(b))

    return [average_b, average_g, average_r]

def background_subtractor(frame1, average_b1, average_g1, average_r1, background_thresholder1, from_x, to_x, from_y, to_y):
    testnpArray1 = np.zeros((720, 1280, 3), np.uint8)
    for a in range(from_y, to_y):
        for j in range(from_x, to_x):
            k = frame1[a, j]
            distance = math.sqrt( (average_r1-k[0])**2 + (average_g1-k[1])**2 + (average_b1-k[2])**2 )
            #print(distance) #try threshold of 90
            if(distance >= background_thresholder1):
                testnpArray1[a][j] = (255,255,255)
            else:
                testnpArray1[a][j] = (0, 0, 0)
    return testnpArray1

def synthesis(dxdy_bbox_input, bs_bbox_input, testnpArray):
    actual_bbox = bs_bbox_input
    tracker = None
    #if the two boxes are close enough, trust the background subtractor
    if (abs(dxdy_bbox_input[0]-bs_bbox_input[0]) <= dxdy_bbox_input[2] and abs(dxdy_bbox_input[1]-bs_bbox_input[1]) <= dxdy_bbox_input[3]):
        actual_bbox = bs_bbox_input
    #if the two boxes are separate, trust the dxdy bbox input then need to reset the tracker (assuming pass continues)
    elif (abs(dxdy_bbox_input[0]-bs_bbox_input[0]) > dxdy_bbox_input[2] or abs(dxdy_bbox_input[1]-bs_bbox_input[1]) > dxdy_bbox_input[3]):
        actual_bbox = dxdy_bbox_input
        tracker = cv2.TrackerCSRT_create()
        tracker.init(testnpArray, actual_bbox)
    return (actual_bbox, tracker)

def main():
    filename = input("Enter full path name to video")
    vid_capture = cv2.VideoCapture(filename)
    while(vid_capture.isOpened() == False):
         filename = input("Invalid path. Enter full path name to video")
         vid_capture = cv2.VideoCapture(filename)
    print("Video accepted.")

    start_time = time.time()

    #DECLARE/DEFINE
    start_frame = 493
    end_frame = 514
    i = start_frame
    tracker = cv2.TrackerCSRT_create()
    r = np.array(0)
    g = np.array(0)
    b = np.array(0)
    background_thresholder = 90
    dxdy = (0, 0)

    #BEGIN ANALYSIS
    vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame = vid_capture.read()

    [average_b, average_g, average_r] = averageBGR(frame)
    #print(average_r)
    #print(average_g)
    #print(average_b)

    ## INIT TRACKING FOR 1ST FRAME
    bbox = (850, 85, 20, 10) #initial bounding box
    center = (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2))

    #set boundaries to try apply background subtractor (region should place bbox in the center of a 3x3 grid)
    to_x = frame.shape[1]
    to_y = frame.shape[0]
    from_x = 0
    from_y = 0
    """if (bbox[0] - bbox[2] > 0):
        from_x = bbox[0] - bbox[2]
    if (bbox[1] - bbox[3] > 0):
        from_y = bbox[1] - bbox[3]
    if (bbox[0] + 2*bbox[2] < frame.shape[0]):
        to_x = bbox[0] + 2*bbox[2]
    if (bbox[1] + 2*bbox[3] < frame.shape[1]):
        to_y = bbox[1] + 2*bbox[3]"""

    ## APPLY BACKGROUND THRESHOLD ##
    testnpArray = background_subtractor(frame, average_b, average_g, average_r, background_thresholder, from_x, to_x, from_y, to_y) #0, frame.shape[0], 0, frame.shape[1])
    #cv2.rectangle(testnpArray, (from_x, from_y), (to_x, to_y), (255, 0, 0))
    #cv2.imshow("testnpArray", testnpArray)
    #cv2.waitKey(0)
    elapsed_time = time.time() - start_time
    print(elapsed_time)
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
      testnpArray = background_subtractor(frame, average_b, average_g, average_r, background_thresholder, from_x, to_x, from_y, to_y)
      #cv2.rectangle(testnpArray, (from_x, from_y), (to_x, to_y), (255, 0, 0))
      #cv2.imshow("testnpArray", testnpArray)
      #cv2.waitKey(0)
      if dxdy == (0,0): #base the first delta value off of background subtractor
            success, bbox = tracker.update(testnpArray)
            new_center = (bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2)
            dxdy = tuple(np.subtract(new_center, center))
            center = new_center
            actual_bbox = bbox
      else: #test value of old delta plus position and goal is to weigh this greater than background subtractor
          # or if tracker gives location much more varied than previous delta, than invalidate background subtractor result

            #guess using dxdy
            attempted_center = tuple(np.rint(np.add(center, dxdy)).astype(int))
            attempted_bbox = (int(attempted_center[0]-bbox[2]/2), int(attempted_center[1]-bbox[3]/2), int(bbox[2]), int(bbox[3]))

            #guess using background subtraction
            success, bbox = tracker.update(testnpArray)

            #synthesize results and update dxdy (SYNTHESIS NOT YET IMPLEMENTED)
            (actual_bbox, isNone) = synthesis(attempted_bbox, bbox, frame)
            bbox = actual_bbox
            if(isNone is not None):
                tracker = isNone
            cv2.rectangle(frame, (attempted_bbox[0], attempted_bbox[1]), (attempted_bbox[0]+attempted_bbox[2], attempted_bbox[1]+attempted_bbox[3]), (0, 0, 255))
            cv2.circle(frame, attempted_center, 2, (0, 0, 255))
            new_center = (actual_bbox[0]+actual_bbox[2]/2, actual_bbox[1]+actual_bbox[3]/2)
            dxdy = tuple(np.subtract(new_center, center))
            center = new_center

            #update region that background subtraction should apply
            if(dxdy[0] >= 0):
                to_x = int(actual_bbox[0]+actual_bbox[2]+3*dxdy[0])
                from_x = int(old_bbox[0]-2*old_bbox[2])
            else:
                to_x = int(old_bbox[0]+2*old_bbox[2])
                from_x = int(actual_bbox[0]-actual_bbox[2]+3*dxdy[0])
            if(dxdy[1] >= 0):
                to_y = int(actual_bbox[1]+actual_bbox[3]+3*dxdy[1])
                from_y = int(old_bbox[1]-2*old_bbox[3])
            else:
                to_y = int(old_bbox[1]+2*old_bbox[3])
                from_y = int(actual_bbox[1]-actual_bbox[3]+3*dxdy[1])

      #draw tracked objects
      p1 = (int(actual_bbox[0]), int(actual_bbox[1]))
      p2 = (int(actual_bbox[0] + actual_bbox[2]), int(actual_bbox[1] + actual_bbox[3]))
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
