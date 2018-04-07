from __future__ import print_function

import os
import pickle
import sys
import time

import neat
import cv2
import numpy as np
import math
from time import sleep

sys.path.append('../')
import ball_on_plate_env as env

# load the winner
with open('winner-ff', 'rb') as f:
    c = pickle.load(f)

local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


ballOnPlate = env.BallOnPlate(showGUI=True, randomInitial=True)

net = neat.nn.FeedForwardNetwork.create(c, config)

ref_point = np.array([0., 0.])

t = 0

envInput = [0, 0]
dropDown = False

ballOnPlate.reset()

posOnPlate  = ballOnPlate.intial_pos
prevPosOnPlate = posOnPlate
prev_err    = [0, 0]
integr_err  = 0


while ballOnPlate.time < 20:
    # half of plate circle
    # ref_point = np.array([.5*math.cos(ballOnPlate.time/2), .5*math.sin(ballOnPlate.time/2)])

    # Get error
    err = ref_point - posOnPlate

    speed = (posOnPlate - prevPosOnPlate)/ballOnPlate.dt

    # Process control system
    netInput = np.array([err[0], err[1], posOnPlate[0], posOnPlate[1], 
                        envInput[0], envInput[1], speed[0], speed[1]])
    # print(netInput)
    netOutput = net.activate(netInput)

    ### PID controller
    prop    = netOutput[0]
    diff    = netOutput[1]
    integr  = netOutput[2]

    integr_err += err
    d_err = err - prev_err

    envInput[0] = prop * err[1] + diff * d_err[1] + integr_err[1] * integr
    envInput[0] = -envInput[0]
    envInput[1] = prop * err[0] + diff * d_err[0] + integr_err[0] * integr

    prev_err = err
    prevPosOnPlate = posOnPlate
    ### PID controller

    envInput = np.clip(envInput, -1, 1)

    posOnPlate, isEnd = ballOnPlate.step(envInput)
    if isEnd:
        # Bad penalty as fall
        print('Fallen!')
        break

    sleep(ballOnPlate.dt)

