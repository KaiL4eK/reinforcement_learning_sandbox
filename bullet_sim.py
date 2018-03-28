import pybullet as p
from time import sleep
import math

import matplotlib as mpl 
import matplotlib.pyplot as plt 

import numpy as np

import ball_on_plate_env as env

ballOnPlate = env.BallOnPlate(showGUI=True)



ref_alpha_vals = []
ref_beta_vals  = []
alpha_vals     = []
beta_vals      = []
t_vals         = []

x_vals         = []
y_vals         = []

r_x_vals         = []
r_y_vals         = []

prev_err = [0, 0]
integr_err = 0

ref_point = np.array([.05, .05])
posOnPlate = np.array([0., 0.])
input = [0, 0]


while 1:

    # # For plotting
    # js1 = p.getJointState(bodyUniqueId=plateId, jointIndex=0)
    # js2 = p.getJointState(bodyUniqueId=plateId, jointIndex=1)
    
    # t_vals += [t]
    # alpha_vals += [js1[0]*180/math.pi]
    # beta_vals += [js2[0]*180/math.pi]
    # ref_alpha_vals += [target_alpha]
    # ref_beta_vals += [target_beta]

    sleep(ballOnPlate.dt) # Time in seconds.
    posOnPlate, isEnd = ballOnPlate.step(input)
    if isEnd:
        break

    # ref_point = np.array([.05*math.cos(t), .05*math.sin(t), 0])

    prop = 0.3
    diff = 100
    integr = 0

    err = ref_point - posOnPlate
    integr_err += err
    d_err = err - prev_err

    input[0] = prop * err[1] + diff * d_err[1] + integr_err[1] * integr
    input[0] = -input[0]
    input[1] = prop * err[0] + diff * d_err[0] + integr_err[0] * integr

    prev_err = err


    # x_vals += [ballPosOnPlate[0]]
    # y_vals += [ballPosOnPlate[1]]

    # r_x_vals += [ref_point[0]]
    # r_y_vals += [ref_point[1]]


    key = p.getKeyboardEvents()
    # if p.B3G_RIGHT_ARROW in key:
    #     ref_point += [.001, 0, 0]
    # if p.B3G_LEFT_ARROW in key:
    #     ref_point -= [.001, 0, 0]
    # if p.B3G_UP_ARROW in key:
    #     ref_point += [0, .001, 0]
    # if p.B3G_DOWN_ARROW in key:
    #     ref_point -= [0, .001, 0]
    # if p.B3G_CONTROL in key:
        # posOnPlate, isEnd = ballOnPlate.step([0, 0])
        # if isEnd:
            # break
    #     ref_point = np.array([0., 0, 0])



    # if p.B3G_ALT in key or ballpos[2] < 0:
    #     # plt.plot(t_vals, alpha_vals, color = 'blue', linestyle = 'solid')
    #     # plt.plot(t_vals, beta_vals, color = 'red', linestyle = 'solid')

    #     plt.plot(x_vals, y_vals, color='blue')
    #     plt.plot(r_x_vals, r_y_vals, color='red')

    #     # plt.plot(t_vals, ref_alpha_vals, color = 'blue', linestyle = 'dashed')
    #     # plt.plot(t_vals, ref_beta_vals, color = 'red', linestyle = 'dashed')
    #     plt.show()

    #     exit(1)

