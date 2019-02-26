import cv2
import numpy as np
from numpy.linalg import inv


class KalmanFilter:
    def __init__(self, X, F, Q, Z, H, R, P, B=np.array([0]), M=np.array([0])):
        self.X = X
        self.P = P
        self.F = F
        self.B = B
        self.M = M
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self):
        # Project the state ahead
        self.X = self.F @ self.X + self.B @ self.M
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.X

    def correct(self, Z):
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P = self.P - K @ self.H @ self.P

        return self.X


TITLE = "Kalman Filter"
frame = np.ones((800,800,3),np.uint8)


def mousemove(event, x, y, s, p):
    global frame, current_measurement, current_prediction,calculated,predicted
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
    current_prediction = kalman.predict()

    cmx, cmy = current_measurement[0], current_measurement[1]
    cpx, cpy = current_prediction[0], current_prediction[1]

    frame = np.ones((800,800,3),np.uint8)
    cv2.putText(frame, "Measurement: ({:.1f}, {:.1f})".format(np.float(cmx), np.float(cmy)),
                (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (50, 150, 0))
    cv2.putText(frame, "Kalman: ({:.1f}, {:.1f})".format(np.float(cpx), np.float(cpy)),
                (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255))
    cv2.circle(frame, (cmx, cmy), 10, (50, 150, 0), -1)      # current measured point
    cv2.circle(frame, (cpx, cpy), 10, (0, 0, 255), -1)      # current predicted point

    calculated.append(current_measurement)
    for z in range(len(calculated)-1):
        p1 = (calculated[z][0],calculated[z][1])
        p2 = (calculated[z+1][0],calculated[z+1][1])
        cv2.line(frame, p1, p2, (50,150,0), 1)
    predicted.append(current_prediction)
    for z in range(len(calculated)-1):
        p1 = (predicted[z][0],predicted[z][1])
        p2 = (predicted[z+1][0],predicted[z+1][1])
        cv2.line(frame, p1, p2, (0,0,255), 1)
    kalman.correct(current_measurement)

    return

calculated=[]
predicted=[]
cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mousemove)

stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
estimateCovariance = np.eye(stateMatrix.shape[0])
transitionMatrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
measurementStateMatrix = np.zeros((2, 1), np.float32)
observationMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
kalman = KalmanFilter(X=stateMatrix,
                      P=estimateCovariance,
                      F=transitionMatrix,
                      Q=processNoiseCov,
                      Z=measurementStateMatrix,
                      H=observationMatrix,
                      R=measurementNoiseCov)

while True:
    cv2.imshow(TITLE,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
