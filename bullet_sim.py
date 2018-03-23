import pybullet as p
from time import sleep
import math

import matplotlib as mpl 
import matplotlib.pyplot as plt 

import numpy as np

physicsClient = p.connect(p.GUI)

plateId = p.loadURDF("plate.urdf")

ballRadius = 52.2 / 2 / 1000
ballMass = 0.205
ballInertia = 2./5*ballMass*ballRadius*ballRadius

sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius)
ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia]*3, 
                          baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex=-1, 
                          basePosition = [0,.1,.26])

# p.changeDynamics(bodyUniqueId=ballId, linkIndex=-1, lateralFriction=10, spinningFriction=0)

p.setGravity(0,0,-9.81)

stepSize=1/1000.
p.setPhysicsEngineParameter(fixedTimeStep=stepSize)

useRealTimeSimulation = 1

if (useRealTimeSimulation):
    p.setRealTimeSimulation(1)

t=0

# around x
target_alpha = 0
# around y
target_beta  = 0

ref_alpha_vals = []
ref_beta_vals  = []
alpha_vals     = []
beta_vals      = []
t_vals         = []

x_vals         = []
y_vals         = []

r_x_vals         = []
r_y_vals         = []


prev_err = [0, 0, 0]
integr_err = 0

ref_point = np.array([.05, .05, 0])

while 1:
    if (useRealTimeSimulation):
        sleep(stepSize) # Time in seconds.
    else:
        p.stepSimulation()
    
    t += stepSize
    
    pos_prop    = 0.01
    vel_prop    = 1
    vel_max     = 2
    force_max   = 3

    p.setJointMotorControl2(bodyUniqueId=plateId, jointIndex=0, controlMode=p.POSITION_CONTROL, 
                            positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                            targetPosition=target_alpha*math.pi/180, force=force_max)

    p.setJointMotorControl2(bodyUniqueId=plateId, jointIndex=1, controlMode=p.POSITION_CONTROL,
                            positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                            targetPosition=target_beta*math.pi/180, force=force_max)

    # For plotting
    js1 = p.getJointState(bodyUniqueId=plateId, jointIndex=0)
    js2 = p.getJointState(bodyUniqueId=plateId, jointIndex=1)
    
    t_vals += [t]
    alpha_vals += [js1[0]*180/math.pi]
    beta_vals += [js2[0]*180/math.pi]
    ref_alpha_vals += [target_alpha]
    ref_beta_vals += [target_beta]

    ballpos, ballorn = p.getBasePositionAndOrientation(ballId)

    ref_point = np.array([.05*math.cos(t), .05*math.sin(t), 0])

    # Check contact
    cp = p.getContactPoints(ballId, plateId, linkIndexB=1)
    if len(cp) == 1:
        ls = p.getLinkState(bodyUniqueId=plateId, linkIndex=1)
        platePos, plateOrn = ls[0], ls[1]
        
        invPlatePos, invPlateOrn = p.invertTransform(platePos, plateOrn)
        ballPosOnPlate, ballOrnOnPlate = p.multiplyTransforms(invPlatePos, invPlateOrn, ballpos, ballorn)

        prop = 40
        diff = 3000
        integr = 0.01

        err = ref_point - ballPosOnPlate
        integr_err += err

        d_err = err - prev_err

        target_alpha = prop * err[1] + diff * d_err[1] + integr_err[1] * integr
        target_beta  = prop * err[0] + diff * d_err[0] + integr_err[0] * integr

        prev_err = err

        target_alpha = -target_alpha

        x_vals += [ballPosOnPlate[0]]
        y_vals += [ballPosOnPlate[1]]

        r_x_vals += [ref_point[0]]
        r_y_vals += [ref_point[1]]

        target_alpha = np.clip(target_alpha, -20, 20)
        target_beta  = np.clip(target_beta, -20, 20)

    print(ref_point[0:2])

    key = p.getKeyboardEvents()
    if p.B3G_RIGHT_ARROW in key:
        ref_point += [.001, 0, 0]
    if p.B3G_LEFT_ARROW in key:
        ref_point -= [.001, 0, 0]
    if p.B3G_UP_ARROW in key:
        ref_point += [0, .001, 0]
    if p.B3G_DOWN_ARROW in key:
        ref_point -= [0, .001, 0]
    # if p.B3G_CONTROL in key:
    #     ref_point = np.array([0., 0, 0])



    if p.B3G_ALT in key or ballpos[2] < 0:
        # plt.plot(t_vals, alpha_vals, color = 'blue', linestyle = 'solid')
        # plt.plot(t_vals, beta_vals, color = 'red', linestyle = 'solid')

        plt.plot(x_vals, y_vals, color='blue')
        plt.plot(r_x_vals, r_y_vals, color='red')

        # plt.plot(t_vals, ref_alpha_vals, color = 'blue', linestyle = 'dashed')
        # plt.plot(t_vals, ref_beta_vals, color = 'red', linestyle = 'dashed')
        plt.show()

        exit(1)

