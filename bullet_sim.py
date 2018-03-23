import pybullet as p
from time import sleep
import math

import matplotlib as mpl 
import matplotlib.pyplot as plt 

import numpy as np

physicsClient = p.connect(p.GUI)

planeId = p.loadURDF("plane.urdf")
plateId = p.loadURDF("plate.urdf")

# VisualShapeId = -1

ballRadius = 52.2 / 2 / 1000
ballMass = 0.205
ballInertia = 2./5*ballMass*ballRadius*ballRadius

# boxExtents = [0.19 / 2, 0.25 / 2, 0.0052 / 2]
# boxMass = 0.415
# boxInertia = [4/12.*(boxExtents[1]*boxExtents[1] + boxExtents[2]*boxExtents[2]),
#             4/12.*(boxExtents[0]*boxExtents[0] + boxExtents[2]*boxExtents[2]),
#             4/12.*(boxExtents[0]*boxExtents[0] + boxExtents[1]*boxExtents[1])]

# linkPositions=[[0,0,0.11],
#              [0,0,0.11]]
# indices=[0, 0]
# jointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]

sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius)
ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia]*3, 
                          baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex=-1, 
                          basePosition = [0,.1,.26])

# baseCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.3])

# plateCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=boxExtents)

# baseId = p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0], 
#                           baseCollisionShapeIndex=baseCollisionShapeId, baseVisualShapeIndex=VisualShapeId, 
#                           basePosition = [0,0,0.3],
#                           )



# Read this: http://alexanderfabisch.github.io/pybullet.html

# for i in range(p.getNumJoints(plateId)):
    # print(p.getJointInfo(plateId, i))

    # p.setJointMotorControl2(bodyUniqueId=plateId, jointIndex=i, controlMode=p.POSITION_CONTROL,
    #                         targetPosition=10*3.14/180, force=10)


# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
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

prev_err = [0, 0, 0]

while 1:
    if (useRealTimeSimulation):
        sleep(stepSize) # Time in seconds.
    else:
        p.stepSimulation()
    
    t += stepSize
    
    pos_prop = 0.1
    vel_prop = 1
    vel_max = 0
    force_max = 10

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

    # Check contact
    cp = p.getContactPoints(ballId, plateId, linkIndexB=1)
    if len(cp) == 1:
        ballpos, ballorn = p.getBasePositionAndOrientation(ballId)

        ls = p.getLinkState(bodyUniqueId=plateId, linkIndex=1)
        platePos, plateOrn = ls[0], ls[1]
        
        invPlatePos, invPlateOrn = p.invertTransform(platePos, plateOrn)
        ballPosOnPlate, ballOrnOnPlate = p.multiplyTransforms(invPlatePos, invPlateOrn, ballpos, ballorn)

        prop = 60
        diff = 1000

        ref_point = np.array([0, 0, 0])

        err = ref_point - ballPosOnPlate
        d_err = err - prev_err

        target_alpha = prop * err[1] + diff * d_err[1] 
        target_beta  = prop * err[0] + diff * d_err[0]

        prev_err = err

        target_alpha = -target_alpha

        target_alpha = np.clip(target_alpha, -20, 20)
        target_beta  = np.clip(target_beta, -20, 20)
    # else:
    #     target_alpha = 0
    #     target_beta  = 0

        # prev_ball_pos = [0, 0]

        # if t >= 3:
        #     plt.plot(t_vals, alpha_vals, color = 'blue', linestyle = 'solid')
        #     plt.plot(t_vals, beta_vals, color = 'red', linestyle = 'solid')

        #     plt.plot(t_vals, ref_alpha_vals, color = 'blue', linestyle = 'dashed')
        #     plt.plot(t_vals, ref_beta_vals, color = 'red', linestyle = 'dashed')
        #     plt.show()

        #     exit(1)

    # key = p.getKeyboardEvents()
    # if p.B3G_RIGHT_ARROW in key:
    #     target_beta = 30
    # if p.B3G_LEFT_ARROW in key:
    #     target_beta = -30
    # if p.B3G_UP_ARROW in key:
    #     target_alpha = -30
    # if p.B3G_DOWN_ARROW in key:
    #     target_alpha = 30
    # if p.B3G_CONTROL in key:
    #     target_alpha = 0
    #     target_beta = 0
    # if p.B3G_ALT in key:
    #     plt.plot(t_vals, alpha_vals, color = 'blue', linestyle = 'solid')
    #     plt.plot(t_vals, ref_vals, color = 'red', linestyle = 'solid')
    #     plt.show()

    #     exit(1)


    # cp = p.getContactPoints( plateId, ballId )
    # if len(cp) > 0:
        # print(cp[0][5])

    # print( p.getBasePositionAndOrientation(plateId) )

    # cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    # print(cubePos,cubeOrn)
