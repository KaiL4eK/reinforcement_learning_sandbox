import os
import sys

import pickle
import time

import neat
import cv2
import ball_on_plate_env as env

simulation_seconds = 20.

################################

from time import sleep
import math
import numpy as np

################################

ref_point = np.array([0., 0.])


def eval_genome(genome, config):
    ballOnPlate = env.BallOnPlate(showGUI=False, randomInitial=False)
    
    net = neat.ctrnn.CTRNN.create(genome, config, ballOnPlate.dt)

    cost = 100

    intial_positions = [[0  , 0  ],
                        [0.4, 0.3],
                        [0.2, 0.5]]

    for i in range(len(intial_positions)):

        net.reset()

        envInput = [0, 0]
        result = 0
        dropDown = False

        ballOnPlate.intial_pos = np.array(intial_positions[i])
        ballOnPlate.reset()

        posOnPlate  = ballOnPlate.intial_pos
        prev_err    = [0, 0]
        integr_err  = 0

        while ballOnPlate.time < simulation_seconds:
            # half of plate circle
            ref_point = np.array([.5*math.cos(ballOnPlate.time/2), .5*math.sin(ballOnPlate.time/2)])

            # Get error
            err = ref_point - posOnPlate
            result -= (err[0] * err[0] + err[1] * err[1]) / 200.

            # Process control system
            netInput = np.array([err[0] / 2, err[1] / 2, posOnPlate[0], posOnPlate[1], envInput[0], envInput[1]])
            # print(netInput)
            netOutput = net.advance(netInput, ballOnPlate.dt, ballOnPlate.dt)

            ### PID controller
            prop    = netOutput[0] * 500
            diff    = netOutput[1] * 100
            integr  = netOutput[2] * 0.1

            integr_err += err
            d_err = err - prev_err

            envInput[0] = prop * err[1] + diff * d_err[1] + integr_err[1] * integr
            envInput[0] = -envInput[0]
            envInput[1] = prop * err[0] + diff * d_err[0] + integr_err[0] * integr

            prev_err = err
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

        cost = min(current_cost, cost)

    ballOnPlate.close()
    return cost
        
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
