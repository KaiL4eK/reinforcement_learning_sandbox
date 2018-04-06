import os
import sys

import pickle
import time

import neat

sys.path.append('../')
import ball_on_plate_env as env

simulation_seconds = 10.

################################

from time import sleep
import math
import numpy as np

################################

def eval_genome(genome, config):
    ballOnPlate = env.BallOnPlate(showGUI=False, randomInitial=False)

    net = neat.ctrnn.CTRNN.create(genome, config, ballOnPlate.dt)

    cost = 1e5

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

        net.reset()

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
            # elif i == 5:
                # ref_point = np.array([.5*math.cos(ballOnPlate.time), .5*math.sin(ballOnPlate.time)])

            # Get error
            err = ref_point - posOnPlate
            result -= (err[0] * err[0] + err[1] * err[1]) * (ballOnPlate.time + 1) / 100.

            speed = (posOnPlate - prevPosOnPlate)/ballOnPlate.dt

            # Process control system
            netInput = np.array([err[0], err[1], posOnPlate[0], posOnPlate[1], 
                                envInput[0], envInput[1], speed[0], speed[1]])
            # print(netInput)
            netOutput = net.advance(netInput, ballOnPlate.dt, ballOnPlate.dt)

            envInput = netOutput

            envInput = np.clip(envInput, -1, 1)
            
            prevPosOnPlate = posOnPlate

            posOnPlate, isEnd = ballOnPlate.step(envInput)
            if isEnd:
                # Bad penalty as fall
                dropDown = True
                break
            # sleep(ballOnPlate.dt)


        if dropDown:
            current_cost = (ballOnPlate.time + result) / simulation_seconds * 100. - 1e4
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
