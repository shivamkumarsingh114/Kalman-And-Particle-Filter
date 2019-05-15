import numpy as np
import cv2
import sys
import math
import argparse

class Condensation():

    """
    Parameters:
      DP         - dimension of the dynamical vector
      MP         - dimension of the measurement vector
      SamplesNum - number of samples in sample set used in algorithm
    """
    def __init__(self, dimDP, dimMP, samplesNum):

        if(dimDP < 0 or dimMP < 0 or samplesNum < 0):
            raise ValueError("Parameters out of range")

        self.SamplesNum = samplesNum
        self.DP = dimDP
        self.MP = dimMP
        self.flSamples =[]
        self.flNewSamples = np.empty([self.SamplesNum, self.DP], dtype=float)
        self.flConfidence = []
        self.flCumulative = np.zeros(self.SamplesNum, dtype=float)
        self.DynamMatr = []
        self.State = np.zeros(self.DP, dtype=float)
        self.lowBound=np.empty(self.DP, dtype=float)
        self.uppBound=np.empty(self.DP, dtype=float)


    # Name: cvConDensInitSampleSet
    # Initializing for the Condensation algorithm
    # Parameters:
    #   conDens     - pointer to CvConDensation structure
    #   lowerBound  - vector of lower bounds used to random update
    #                   of sample set
    #   upperBound  - vector of upper bounds used to random update
    #                   of sample set
    #
    def cvConDensInitSampleSet(self, lowerBound, upperBound):
        prob = 1.0/self.SamplesNum

        if((lowerBound is None) or (upperBound is None)):
            raise ValueError("Lower/Upper Bound")

        self.lowBound = lowerBound
        self.uppBound = upperBound

        # Generating the samples
        for j in range(self.SamplesNum):
            valTmp = np.zeros(self.DP, dtype=float)
            for i in range(self.DP):
                valTmp[i] = np.random \
                    .uniform(lowerBound[i],upperBound[i])
            self.flSamples.append(valTmp)
            self.flConfidence.append(prob)

    # Name:    cvConDensUpdateByTime
    # Performing Time Update routine for ConDensation algorithm
    # Parameters:
    def cvConDensUpdateByTime(self):
        valSum  = 0
        #Sets Temp To Zero
        self.Temp = np.zeros(self.DP, dtype=float)

        #Calculating the Mean
        for i in range(self.SamplesNum):
            self.State = np.multiply(self.flSamples[i], self.flConfidence[i])
            self.Temp = np.add(self.Temp, self.State)
            valSum += self.flConfidence[i]
            self.flCumulative[i] = valSum

        #Taking the new vector from transformation of mean by dynamics matrix
        self.Temp = np.multiply(self.Temp, (1.0/valSum))
        for i in range(self.DP):
            self.State[i] = np.sum(np.multiply(self.DynamMatr[i],
                                               self.Temp[i]))

        valSum = valSum / float(self.SamplesNum)

        #Updating the set of random samples
        for i in range(self.SamplesNum):
            j = 0
            while (self.flCumulative[j] <= float(i)*valSum and
                j < self.SamplesNum -1):
                j += 1
            self.flNewSamples[i] = self.flSamples[j]

        lowerBound = np.empty(self.DP, dtype=float)
        upperBound = np.empty(self.DP, dtype=float)
        for j in range(self.DP):
            lowerBound[j] = (self.lowBound[j] - self.uppBound[j])/5.0
            upperBound[j] = (self.uppBound[j] - self.lowBound[j])/5.0

        #Adding the random-generated vector to every vector in sample set
        RandomSample = np.empty(self.DP, dtype=float)
        for i in range(self.SamplesNum):
            for j in range(self.DP):
                RandomSample[j] = np.random.uniform(lowerBound[j],
                                                    upperBound[j])
                self.flSamples[i][j] = np.sum(np.multiply(self.DynamMatr[j],
                                               self.flNewSamples[i][j]))

            self.flSamples[i] = np.add(self.flSamples[i], RandomSample)


keep_processing = True;

selection_in_progress = False; # support interactive region selection
fullscreen = False; # run in fullscreen mode

# parse command line arguments for camera ID or video file

parser = argparse.ArgumentParser(description='Perform Condensation (particle filter) tracking on an incoming camera image')
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

#####################################################################

# select a region using the mouse

boxes = [];
current_mouse_position = np.ones(2, dtype=np.int32);

def on_mouse(event, x, y, flags, params):

    global boxes;
    global selection_in_progress;

    current_mouse_position[0] = x;
    current_mouse_position[1] = y;

    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = [];
        # print 'Start Mouse Position: '+str(x)+', '+str(y)
        sbox = [x, y];
        selection_in_progress = True;
        boxes.append(sbox);

    elif event == cv2.EVENT_LBUTTONUP:
        # print 'End Mouse Position: '+str(x)+', '+str(y)
        ebox = [x, y];
        selection_in_progress = False;
        boxes.append(ebox);
#####################################################################

# return centre of a set of points representing a rectangle

def center(points):
    x = np.float32((points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0)
    y = np.float32((points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0)
    return np.array([np.float32(x), np.float32(y)], np.float32)

#####################################################################

# this function is called as a call-back everytime the trackbar is moved
# (here we just do nothing)

def nothing(x):
    pass

#####################################################################

# draw a cross on the specified image at location, colour, size (d)
# specified

def drawCross(img, center, color, d):
    #On error change cv2.CV_AA for cv2.LINE_AA
    # (for differents versions of OpenCV)
    cv2.line(img, (center[0] - d, center[1] - d), \
             (center[0] + d, center[1] + d), color, 2, cv2.LINE_AA, 0)
    cv2.line(img, (center[0] + d, center[1] - d), \
             (center[0]- d, center[1] + d), color, 2, cv2.LINE_AA, 0)

#####################################################################
# define video capture object

cap = cv2.VideoCapture();

# define display window name

windowName = "Condensation Tracking"; # window name
windowName2 = "Hue histogram back projection"; # window name
windowNameSelection = "initial selected region";

# init Condensation object

dimensions = 2          # number of parameters for tracking
nParticles = 100        # number of particles to use
xRange = 640.0          # image width
yRange = 480.0          # image hieght
LB = [0.0, 0.0]         # lower bounds on sampling
UB = [xRange, yRange]   # upper bounds on sampling
tracker = Condensation(dimensions, dimensions, nParticles)
tracker.cvConDensInitSampleSet(LB, UB)
tracker.DynamMatr = [[1.0, 0.0],[0.0, 1.0]]

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (((args.video_file) and (cap.open(str(args.video_file))))
    or (cap.open(args.camera_to_use))):

    # create window by name (note flags for resizable or not)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL);
    cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL);

    # set sliders for HSV selection thresholds

    s_lower = 60;
    cv2.createTrackbar("s lower", windowName2, s_lower, 255, nothing);
    s_upper = 255;
    cv2.createTrackbar("s upper", windowName2, s_upper, 255, nothing);
    v_lower = 32;
    cv2.createTrackbar("v lower", windowName2, v_lower, 255, nothing);
    v_upper = 255;
    cv2.createTrackbar("v upper", windowName2, v_upper, 255, nothing);

    # set a mouse callback

    cv2.setMouseCallback(windowName, on_mouse, 0);
    cropped = False;

    print("\nObservation in image: BLUE");
    print("Prediction from Condensation Tracker: GREEN\n");

    # Setup the termination criteria for search, either 10 iteration or
    # move by at least 1 pixel pos. difference
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while (keep_processing):
        # if video file/ cam successfully open then read frame

        if (cap.isOpened):
            ret, frame = cap.read();
        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # get parameters from track bars

        s_lower = cv2.getTrackbarPos("s lower", windowName2);
        s_upper = cv2.getTrackbarPos("s upper", windowName2);
        v_lower = cv2.getTrackbarPos("v lower", windowName2);
        v_upper = cv2.getTrackbarPos("v upper", windowName2);

        # select region using the mouse and display it

        if (len(boxes) > 1) and (boxes[0][1] < boxes[1][1]) and (boxes[0][0] < boxes[1][0]):
            crop = frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()

            h, w, c = crop.shape;   # size of template
            if (h > 0) and (w > 0):
                cropped = True;

                # convert region to HSV

                hsv_crop =  cv2.cvtColor(crop, cv2.COLOR_BGR2HSV);

                # select all Hue and Sat. values (0-> 180) but eliminate values with very low
                # saturation or value (due to lack of useful colour information)

                mask = cv2.inRange(hsv_crop, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))));
                # mask = cv2.inRange(hsv_crop, np.array((0., 60.,32.)), np.array((180.,255.,255.)));

                # construct a histogram of hue and saturation values and normalize it

                crop_hist = cv2.calcHist([hsv_crop],[0, 1],mask,[180, 255],[0,180, 0, 255]);
                cv2.normalize(crop_hist,crop_hist,0,255,cv2.NORM_MINMAX);

                # set intial position of object

                track_window = (boxes[0][0],boxes[0][1],boxes[1][0] - boxes[0][0],boxes[1][1] - boxes[0][1]);

                cv2.imshow(windowNameSelection,crop);

            # reset list of boxes

            boxes = [];

        # interactive display of selection box

        if (selection_in_progress):
            top_left = (boxes[0][0], boxes[0][1]);
            bottom_right = (current_mouse_position[0], current_mouse_position[1]);
            cv2.rectangle(frame,top_left, bottom_right, (0,255,0), 2);

        # if we have a selected region

        if (cropped):

            # convert incoming image to HSV

            img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);

            img_bproject = cv2.calcBackProject([img_hsv],[0,1],crop_hist,[0,180,0,255],1);
            cv2.imshow(windowName2,img_bproject);

            # apply camshift to predict new location (observation)
            # basic HSV histogram comparision with adaptive window size
            # see : http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html
            ret, track_window = cv2.CamShift(img_bproject, track_window, term_crit);

            # draw observation on image
            x,y,w,h = track_window;
            frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2);

            # extract centre of this observation as points

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            pts = center(pts)

            # use to update Condensation tracker

            for hypothesis in range(tracker.SamplesNum):

                # calculate the confidence based on the observations

                diffX = (pts[0] - tracker.flSamples[hypothesis][0]) / xRange;
                diffY = (pts[1] - tracker.flSamples[hypothesis][1]) / yRange;

                # calculate a confidence / probabilty based on this observation
                # and our motion model - here we assume motion in any direction
                # is equally likely - hence probabilty is 1 / (distance between
                # observation and sample hypothesis)

                # this could/should be updated to a relevant motion model for
                # the scenario in use; see also Condensation.py object code

                tracker.flConfidence[hypothesis] = 1.0/(np.sqrt(np.power(diffX,2) + \
                                                    np.power(diffY,2)))

            tracker.cvConDensUpdateByTime();

            # get new Condensation tracker prediction

            predictionF = tracker.State;
            prediction = [int(s) for s in predictionF]

            # draw predicton on image

            frame = cv2.rectangle(frame, (prediction[0]-int(0.5*w),prediction[1]-int(0.5*h)), (prediction[0]+int(0.5*w),prediction[1]+int(0.5*h)), (0,255,0),2);

            # draw all the tracker hypothesis samples on the image

            for j in range(len(tracker.flSamples)):
                posNew = [int(s) for s in tracker.flSamples[j]]
                drawCross(frame, posNew, (255, 255, 0), 2)

        else:

            # before we have cropped anything show the mask we are using
            # for the S and V components of the HSV image

            img_hsv =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV);

            # select all Hue values (0-> 180) but eliminate values with very low
            # saturation or value (due to lack of useful colour information)

            mask = cv2.inRange(img_hsv, np.array((0., float(s_lower),float(v_lower))), np.array((180.,float(s_upper),float(v_upper))));

            cv2.imshow(windowName2,mask);

        # display image

        cv2.imshow(windowName,frame);
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN & fullscreen);

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
        # here we use a wait time in ms. that takes account of processing time already used in the loop

        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(2, 40 - int(math.ceil(stop_t)))) & 0xFF;

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit / press "f" for fullscreen display

        if (key == ord('q')):
            keep_processing = False;
        elif (key == ord('f')):
            fullscreen = not(fullscreen);

    # close all windows

    cv2.destroyAllWindows()

else:
    print("No video file specified or camera connected.");
