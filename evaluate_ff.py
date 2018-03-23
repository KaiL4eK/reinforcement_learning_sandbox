import os
import sys

import pickle
import time

import neat
import cv2


simulation_seconds = 20

################################

import pybullet as p
from time import sleep
import math
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np

################################

show_gui = False

if show_gui:
    physicsClient = p.connect(p.GUI)

def eval_genome(genome, config):
    if not show_gui:
        physicsClient = p.connect(p.DIRECT)
    p.resetSimulation()

    plateId = p.loadURDF("plate.urdf")

    ballRadius = 52.2 / 2 / 1000
    ballMass = 0.205
    ballInertia = 2./5*ballMass*ballRadius*ballRadius

    sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius)
    ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia]*3, 
                              baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex=-1, 
                              basePosition = [0,.1,.26])

    p.setGravity(0,0,-9.81)

    stepSize=1/1000.
    p.setPhysicsEngineParameter(fixedTimeStep=stepSize)

    ref_point = np.array([0., 0., 0.])

    t = 0

    # around x
    target_alpha = 0
    # around y
    target_beta  = 0

    pos_prop    = 0.01
    vel_prop    = 1
    vel_max     = 2
    force_max   = 3

    not_touched = True

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    result = 0

    while t < simulation_seconds:
        if not_touched and len(p.getContactPoints(ballId, plateId, linkIndexB=1)) == 0:
            pass
        else:
            not_touched = False

            # Calc input
            # ref_point = np.array([.05*math.cos(t), .05*math.sin(t), 0])

            # Get position
            ballpos, ballorn = p.getBasePositionAndOrientation(ballId)

            ls = p.getLinkState(bodyUniqueId=plateId, linkIndex=1)
            platePos, plateOrn = ls[0], ls[1]
            
            invPlatePos, invPlateOrn = p.invertTransform(platePos, plateOrn)
            ballPosOnPlate, ballOrnOnPlate = p.multiplyTransforms(invPlatePos, invPlateOrn, ballpos, ballorn)

            # Get error
            err = ref_point - ballPosOnPlate
            result -= (err[0] * err[0] + err[1] * err[1]) * 1e3

            if ballpos[2] < .1:
                if not show_gui:
                    p.disconnect()
                return result / 20000. * 100.

            # Process control system
            inputs = np.array([ref_point[0] / 0.2 - 1., ref_point[1] / 0.2 - 1., ballPosOnPlate[0] / 0.2 - 1., ballPosOnPlate[1] / 0.2 - 1.])

            action = net.activate(inputs)

            # Get control
            target_alpha = action[0] * 20
            target_beta  = action[1] * 20

            p.setJointMotorControl2(bodyUniqueId=plateId, jointIndex=0, controlMode=p.POSITION_CONTROL, 
                                    positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                    targetPosition=target_alpha*math.pi/180, force=force_max)

            p.setJointMotorControl2(bodyUniqueId=plateId, jointIndex=1, controlMode=p.POSITION_CONTROL,
                                    positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                    targetPosition=target_beta*math.pi/180, force=force_max)

        p.stepSimulation()
        
        t += stepSize
        

    result += t * 10000
    
    if not show_gui:
        p.disconnect()
        
    return result / 20000. * 100.

def eval_genomes(genomes, config):

    # img = sim_map.get_image(resol)

    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

    # cv2.imshow('0', cv2.flip(img, 0))
    # cv2.waitKey(30)
