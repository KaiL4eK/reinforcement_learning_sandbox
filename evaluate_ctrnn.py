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

    # for i in range(2):

    net.reset()

    envInput = [0, 0]
    result = 0
    dropDown = False

    ballOnPlate.reset()

    while ballOnPlate.time < simulation_seconds:
        # half of plate circle
        ref_point = np.array([.5*math.cos(ballOnPlate.time/2), .5*math.sin(ballOnPlate.time/2)])

        # if ballOnPlate.is_contacted():
        #     contact = 1
        # else:
        #     contact = 0 

        posOnPlate, isEnd = ballOnPlate.step(envInput)
        if isEnd:
            # Bad penalty as fall
            dropDown = True
            break

        # Get error
        err = ref_point - posOnPlate
        result -= (err[0] * err[0] + err[1] * err[1]) / 200.

        # Process control system
        netInput = np.array([err[0] / 2, err[1] / 2, posOnPlate[0], posOnPlate[1]])
        envInput = net.advance(netInput, ballOnPlate.dt, ballOnPlate.dt)

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
