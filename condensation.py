import numpy as np
import cv2
import numpy as np

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

class MouseGUI():

    def __init__(self, parent):

        self.mouse_info = (-1, -1)

        self.mouseV = []
        self.calculated = []

        self.counter = -1

        self.initialize()

    def initialise_defaults(self):
        """
        This routine fills in the data structures with default constant
        values.
        """

        dim = 2
        nParticles = 50
        self.xRange = 650.0
        self.yRange = 650.0
        LB = [0.0, 0.0]
        UB = [self.xRange, self.yRange]
        self.model = Condensation(dim, dim, nParticles)
        self.model.cvConDensInitSampleSet(LB, UB)
        self.model.DynamMatr = [[1.0, 0.0],
                     [0.0, 1.0]]

    def initialize(self):

        #Initialize model specific conditions
        self.initialise_defaults()

        # Windows Init
        cv2.namedWindow('Particle Filter')
        cv2.setMouseCallback('Particle Filter', self.on_mouse)

        img = np.zeros((650, 650 ,3), np.uint8)*255

        countPts = 0

        while True:

            # Clear all
            img[:,:] = (0,0,0)

            sizeVec = len(self.mouseV) - 1

            if(sizeVec == -1):
                cv2.imshow('Particle Filter', img)
                cv2.waitKey(30)
                continue

            for i in range(sizeVec):
                cv2.line(img, self.mouseV[i], self.mouseV[i+1], \
                         (50,150,0), 1)

            mean = self.model.State
            meanInt = [int(s) for s in mean]

            self.calculated.append(meanInt)

            for k in range(len(self.calculated)-1):
                p1 = (self.calculated[k][0], self.calculated[k][1])
                p2 = (self.calculated[k+1][0], self.calculated[k+1][1])
                if p1[0]!=0 and p1[1]!=0:
                    cv2.line(img, p1, p2, (0,0,255), 1)

            if int(self.counter) == int(0):
                pts = self.mouseV[sizeVec]
                for z in range(self.model.SamplesNum):

                    #Calculate the confidence based on the observations
                    diffX = (pts[0] - self.model.flSamples[z][0])/self.xRange
                    diffY = (pts[1] - self.model.flSamples[z][1])/self.yRange
                    self.model.flConfidence[z] = 1.0/(np.sqrt(np.power(diffX,2) + \
                                               np.power(diffY,2)))


                # Updates
                self.model.cvConDensUpdateByTime()
                self.update_after_iterating(img)


                #Not required, (max n points)
                if countPts > 100:
                    break

            countPts+=1

            self.counter += 1

            cv2.imshow('Particle Filter', img)

            key = cv2.waitKey(1000)

            if key == ord('q'):
                break

        cv2.destroyAllWindows()


    def on_mouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_MOUSEMOVE:
            self.last_mouse = self.mouse_info
            self.mouse_info = (x, y)
            self.counter = 0
            self.mouseV.append((x,y))


    def drawCross(self, img, center, color, d):
        cv2.line(img, (center[0] - d, center[1] - d), \
                 (center[0] + d, center[1] + d), color, 2, cv2.LINE_AA, 0)
        cv2.line(img, (center[0] + d, center[1] - d), \
                 (center[0]- d, center[1] + d), color, 2, cv2.LINE_AA, 0)

    def update_after_iterating(self, img):

        mean = self.model.State
        meanInt = [int(s) for s in mean]

        for j in range(len(self.model.flSamples)):
            posNew = [int(s) for s in self.model.flSamples[j]]
            self.drawCross(img, posNew, (255, 255, 0), 2)

        self.calculated.append(meanInt)

        sizeVec = len(self.mouseV) - 1
        print("Measurement: %d %d" % (meanInt[0], meanInt[1]))
        print("Particle Filter: %d %d" % (self.mouseV[sizeVec][0], self.mouseV[sizeVec][1]))
        print( '+++++++++++++++')
        self.drawCross(img, meanInt, (255, 0, 255), 2)
        cv2.putText(img, "Measurement: ({:.1f}, {:.1f})".format(np.float(meanInt[0]), np.float(meanInt[1])),
                    (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
        cv2.putText(img, "Particle Filter: ({:.1f}, {:.1f})".format(np.float(self.mouseV[sizeVec][0]),+\
                                            np.float(self.mouseV[sizeVec][1])),
                    (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))

if __name__ == "__main__":
    app = MouseGUI(None)
