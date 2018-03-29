import pybullet as p
import math as m
import numpy as np
import random as rand

class BallOnPlate:

    def __init__(self, showGUI=False, randomInitial=False):
        self.dt = 1/1000.
        self.controlAngleLimit = 20
        self.plateSize = 0.2
        self.randomInitial = randomInitial

        self.intial_pos = np.array([0., 0.])

        # Now work with simulator
        if showGUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.reset()

    # Input - desired angles [-1; 1]
    def step(self, input):

        input = np.clip(input, -1, 1)
        self.angleTargets = np.array(input) * self.controlAngleLimit * m.pi / 180

        pos_prop    = 0.01
        vel_prop    = 1
        vel_max     = 2
        force_max   = 3

        p.setJointMotorControl2(bodyUniqueId=self.plateId, jointIndex=0, controlMode=p.POSITION_CONTROL, 
                                positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                targetPosition=self.angleTargets[0], force=force_max)

        p.setJointMotorControl2(bodyUniqueId=self.plateId, jointIndex=1, controlMode=p.POSITION_CONTROL,
                                positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                targetPosition=self.angleTargets[1], force=force_max)

        p.stepSimulation()
        self.time += self.dt

        ballpos, ballorn = p.getBasePositionAndOrientation(self.ballId)
        
        ls = p.getLinkState(bodyUniqueId=self.plateId, linkIndex=1)
        platePos, plateOrn = ls[0], ls[1]
        
        invPlatePos, invPlateOrn = p.invertTransform(platePos, plateOrn)
        ballPosOnPlate, ballOrnOnPlate = p.multiplyTransforms(invPlatePos, invPlateOrn, ballpos, ballorn)
        ballPosOnPlate = np.array(ballPosOnPlate)

        # [x, y] on plate in range [-1; 1]
        self.ballPosition = ballPosOnPlate[0:2] / self.plateSize
        self.ballHeight = ballpos[2]

        return self.ballPosition, self._is_end()

    def is_contacted(self):
        return len(p.getContactPoints(self.ballId, self.plateId, linkIndexB=1)) != 0

    def _is_fallen(self):
        return self.ballHeight < .1

    def _is_end(self):
        return (self._is_fallen() and not self.is_contacted())

    def reset(self):
        self.time           = 0

        if self.randomInitial:
            self.intial_pos = [(rand.random() * 2 - 1) / 2, (rand.random() * 2 - 1) / 2]

        self.ballHeight     = .28
        self.ballPosition   = self.intial_pos * self.plateSize

        # Alpha, Beta
        self.angleTargets = [0, 0]

        p.resetSimulation()

        p.setGravity(0,0,-9.81)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

        ballRadius = 52.2 / 2 / 1000
        ballMass = 0.205
        ballInertia = 2./5*ballMass*ballRadius*ballRadius

        self.plateId = p.loadURDF("plate.urdf")

        sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius)
        self.ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia]*3, 
                                  baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex=-1, 
                                  basePosition = [self.ballPosition[0], self.ballPosition[1], self.ballHeight])

    def close(self):
        p.disconnect()
