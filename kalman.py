import numpy as np
from numpy.linalg import inv
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Simple Kalman filter

Initialise the filter
Args:
    X: State estimate
    P: Estimate covariance
    A: State transition model
    B: Control matrix
    M: Control vector
    Q: Process noise covariance
    Z: Measurement of the state X
    H: Observation model
    R: Observation noise covariance
"""

x_observations = np.array([0])
y_observations = np.array([0])
vx_observations = np.array([4])
vy_observations = np.array([-4])
z = np.c_[x_observations,y_observations, vx_observations,vy_observations]
# Initial Conditions
a = [-5,1]  # Acceleration
t = 0.1  # Difference in time

# Process / Estimation Errors
error_est_x = 20 # Uncertainty in prediction
error_est_y = 10
error_est_vx = 5
error_est_vy = 5

# Observation/measurement Errors
error_obs_x = 25
error_obs_y = 15  # Uncertainty in the measurement
error_obs_vx = 6
error_obs_vy = 4

def prediction2d(x,y,vx,vy, t, a):
    A = np.array([[1,0,t,0],
                  [0, 1,0,t],
                  [0,0,1,0],[0,0,0,1]])
    X = np.array([[x,y,
                  vx,vy]])
    B = np.array([0.5 * t ** 2,0.5 * t ** 2,t,t])
    a=np.array([a,a]).reshape(1,4)
    X=np.array(X).reshape(1,4)
    X_prime = A.dot(X.T)+ (B*a).T
    print(X_prime)
    return X_prime


def covariance2d(sigmax, sigmay,sigmavx, sigmavy,):
    sigma=np.array([sigmax,sigmay,sigmavx,sigmavy])
    cov_matrix = sigma.dot(np.eye(4,4))
    cov_matrix=cov_matrix*np.eye(4,4)
    return (cov_matrix)


# Initial Estimation/Process Covariance Matrix
P = covariance2d(error_est_x, error_est_y,
                error_est_vx, error_est_vy)
A = np.array([[1,0,t,0],
              [0, 1,0,t],
              [0,0,1,0],[0,0,0,1]])

# Initial State Matrix
X = np.array([[z[0][0]],
              [z[0][1]],
              [z[0][2]],
              [z[0][3]]])
predicted_values=[]
measured_values=[]
kalman_values=[]
n = len(z[0])
iter=50
for i in range(0,iter):
    a[0]=np.add(a[0],(1/4))
    a[1]=np.add(a[1],-(1/4))
    X = prediction2d(X[0][0], X[1][0] ,X[2][0], X[3][0], t, a)
    predicted_values.append(X)

    # set off-diagonal terms to 0.
    P = np.diag(np.diag(A.dot(P).dot(A.T)))

    # Calculating the Kalman Gain
    H = np.identity(n)
    R = covariance2d(error_obs_x, error_obs_y,
            error_obs_vx, error_obs_vy) #measurement/Observation covariance matrix
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H).dot(inv(S))
    # Reshape the new data into the measurement space.
    data = np.array([[X[0,0] + abs(random.randn(1)[0])], [X[1,0] +\
      abs(random.randn(1)[0])],
      [X[2,0] + abs(random.randn(1)[0])], [X[3,0] +\
        abs(random.randn(1)[0])]])
    Y = H.dot(data).reshape(n, -1)
    measured_values.append(Y)
    # Update the State Matrix
    # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
    X = X + K.dot(Y -H.dot(X))
    kalman_values.append(X)
    # Update Process Covariance Matrix
    P = (np.identity(len(K)) - K.dot(H)).dot(P)

    if X[0][0]>=10 or X[0][0]<=0: #virtual area of 10*10
        X[2][0]=-X[2][0]
    if X[1][0]>=10 or X[1][0]<=0:
        X[3][0]=-X[3][0]


kalman_values=np.array(kalman_values).reshape(iter,4)
measured_values=np.array(measured_values).reshape(iter,4)
predicted_values=np.array(predicted_values).reshape(iter,4)
kalman_values_x=[kalman_values[i][0] for i in range(0,kalman_values.shape[0])]
kalman_values_y=[kalman_values[i][1] for i in range(0,kalman_values.shape[0])]
kalman_values_vx=[kalman_values[i][2] for i in range(0,kalman_values.shape[0])]
kalman_values_vy=[kalman_values[i][3] for i in range(0,kalman_values.shape[0])]
measured_values_x=[measured_values[i][0] for i in range(0,measured_values.shape[0])]
measured_values_y=[measured_values[i][1] for i in range(0,measured_values.shape[0])]
measured_values_vx=[measured_values[i][2] for i in range(0,measured_values.shape[0])]
measured_values_vy=[measured_values[i][3] for i in range(0,measured_values.shape[0])]
predicted_values_x=[predicted_values[i][0] for i in range(0,predicted_values.shape[0])]
predicted_values_y=[predicted_values[i][1] for i in range(0,predicted_values.shape[0])]
predicted_values_vx=[predicted_values[i][2] for i in range(0,predicted_values.shape[0])]
predicted_values_vy=[predicted_values[i][3] for i in range(0,predicted_values.shape[0])]

kalman_pos= np.sqrt(np.power(kalman_values_x, 2)+np.power(kalman_values_y, 2))
measured_pos= np.sqrt(np.power(measured_values_x, 2)+np.power(measured_values_y, 2))
predicted_pos= np.sqrt(np.power(predicted_values_x, 2)+np.power(predicted_values_y, 2))
plt.plot([i for i in range(0,iter)],kalman_values_x,'r',measured_values_x,'b',predicted_values_x,'g')
plt.xlabel("time")
plt.ylabel("Position in X direction")
plt.legend(("Kalman","Measured","Predicted"))
plt.show()
plt.plot([i for i in range(0,iter)],kalman_values_y,'r',measured_values_y,'b',predicted_values_y,'g')
plt.xlabel("time")
plt.ylabel("Position in Y direction")
plt.legend(("Kalman","Measured","Predicted"))
plt.show()
plt.plot([i for i in range(0,iter)],kalman_values_vx,'r',measured_values_vx,'b',predicted_values_vx,'g')
plt.xlabel("time")
plt.ylabel("Velocity in X direction")
plt.legend(("Kalman","Measured","Predicted"))
plt.show()
plt.plot([i for i in range(0,iter)],kalman_values_vy,'r',measured_values_vy,'b',predicted_values_vy,'g')
plt.xlabel("time")
plt.ylabel("Velocity in Y direction")
plt.legend(("Kalman","Measured","Predicted"))
plt.show()

plt.plot([i for i in range(0,iter)],kalman_pos,'r',measured_pos,'b',predicted_pos,'g')
plt.xlabel("time")
plt.ylabel("Position")
plt.legend(("Kalman","Measured","Predicted"))
plt.show()

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kalman_values_x, kalman_values_y,0, c='gray')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Ball Trajectory observed from Computer Vision System (with Noise)')
plt.show()
