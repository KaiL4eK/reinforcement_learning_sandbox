import os
import sys

import pickle
import time

import neat
import cv2

sys.path.append('../')
import ball_on_plate_env as env

simulation_seconds = 10.

################################

from time import sleep
import math
import numpy as np

################################

ref_point = np.array([0., 0.])


def eval_genome(genome, config):
    ballOnPlate = env.BallOnPlate(showGUI=False, randomInitial=False)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    cost = 100

    # intial_positions = [[0  , 0  ],
    #                     [-0.4, 0.3],
    #                     [-0.5, -0.5],
    #                     [0.7, 0.7]]
    CONST_VALUE = 0.7
    intial_positions = [[CONST_VALUE, CONST_VALUE],
                        [-CONST_VALUE, -CONST_VALUE],
                        [-CONST_VALUE, CONST_VALUE],
                        [CONST_VALUE, -CONST_VALUE],
                        [0., 0.], [0., 0.]]

    reference_positions = [[-CONST_VALUE, -CONST_VALUE],
                           [CONST_VALUE, CONST_VALUE],
                           [CONST_VALUE, -CONST_VALUE],
                           [-CONST_VALUE, CONST_VALUE],
                           [0., 0.], [0., 0.]]

    for i in range(len(intial_positions)):

        envInput = [0, 0]
        result = 0
        dropDown = False

        ballOnPlate.intial_pos  = np.array(intial_positions[i])
        ref_point               = np.array(reference_positions[i])

        posOnPlate = ballOnPlate.reset()
        prevPosOnPlate = posOnPlate

        prev_err    = [0, 0]
        integr_err  = 0

        while ballOnPlate.time < simulation_seconds:
            # half of plate circle
            if i == 4:
                ref_point = np.array([.5*math.cos(ballOnPlate.time/2), .5*math.sin(ballOnPlate.time/2)])
            elif i == 5:
                ref_point = np.array([.5*math.cos(ballOnPlate.time), .5*math.sin(ballOnPlate.time)])

            # Get error
            err = ref_point - posOnPlate
            result -= (err[0] * err[0] + err[1] * err[1]) * (ballOnPlate.time + 1) / 100.

            speed = (posOnPlate - prevPosOnPlate)/ballOnPlate.dt

            # Process control system
            netInput = np.array([err[0] / 2, err[1] / 2, posOnPlate[0], posOnPlate[1], 
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
                dropDown = True
                break
            # sleep(ballOnPlate.dt)


        if dropDown:
            current_cost = (ballOnPlate.time + result) / simulation_seconds * 100. - 100
        else:
            current_cost = (ballOnPlate.time + result) / simulation_seconds * 100.

        # cost += current_cost
        cost = min(current_cost, cost)

    ballOnPlate.close()
    # return cost / len(intial_positions)
    return cost
        
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)