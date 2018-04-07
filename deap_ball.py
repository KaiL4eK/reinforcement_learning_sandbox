#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.


#    example which maximizes the sum of a list of integers
#    each of which can be 0 or 1

import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy
import multiprocessing


################################

import os
import sys

import pickle
import time

import neat

sys.path.append('../')
import ball_on_plate_env as env

simulation_seconds = 10.

from time import sleep
import math
import numpy as np

################################

def eval_genome(genome):
    ballOnPlate = env.BallOnPlate(showGUI=False, randomInitial=False)

    cost = 0

    CONST_VALUE = 0.7
    intial_positions = [[CONST_VALUE, CONST_VALUE],
                        [-CONST_VALUE, -CONST_VALUE],
                        [-CONST_VALUE, CONST_VALUE],
                        [CONST_VALUE, -CONST_VALUE],
                        [0., 0.]]

    reference_positions = [[-CONST_VALUE, -CONST_VALUE],
                           [CONST_VALUE, CONST_VALUE],
                           [CONST_VALUE, -CONST_VALUE],
                           [-CONST_VALUE, CONST_VALUE],
                           [0., 0.]]

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

            # Get error
            err = ref_point - posOnPlate
            result -= (err[0] * err[0] + err[1] * err[1]) * (ballOnPlate.time + 1) / 100.

            ### PID controller
            prop    = genome[0] * 1.
            diff    = genome[1] * 1.
            integr  = genome[2] * 1.

            integr_err += err
            d_err = err - prev_err

            envInput[0] = prop * err[1] + diff * d_err[1] + integr_err[1] * integr
            envInput[0] = -envInput[0]
            envInput[1] = prop * err[0] + diff * d_err[0] + integr_err[0] * integr

            prev_err = err
            ### PID controller

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

        cost += current_cost / float(len(intial_positions))
        # cost = min(current_cost, cost)

    ballOnPlate.close()
    return cost,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
# toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("attr_float", random.random)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, 3)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", eval_genome)

# register the crossover operator
toolbox.register("mate", tools.cxUniform, indpb=0.7)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutGaussian, indpb=0.1, mu=0, sigma=.02)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=7)

#----------

def main():
    random.seed(time.time())

    # Process Pool of 4 workers
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=10000, 
                            stats=stats, halloffame=hof)
    except KeyboardInterrupt:
        print('Try-catch')

    pool.close()
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
    main()
