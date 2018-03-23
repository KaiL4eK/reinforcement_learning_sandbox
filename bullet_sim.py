import pybullet as p
from time import sleep

physicsClient = p.connect(p.GUI)

planeId = p.loadURDF("plane.urdf")


VisualShapeId = -1



ballRadius = 52.2 / 2 / 1000
ballMass = 0.205
ballInertia = 2./5*ballMass*ballRadius*ballRadius

boxExtents = [0.19 / 2, 0.25 / 2, 0.0052 / 2]
boxMass = 0.415
boxInertia = [4/12.*(boxExtents[1]*boxExtents[1] + boxExtents[2]*boxExtents[2]),
			  4/12.*(boxExtents[0]*boxExtents[0] + boxExtents[2]*boxExtents[2]),
			  4/12.*(boxExtents[0]*boxExtents[0] + boxExtents[1]*boxExtents[1])]

linkPositions=[[0,0,0.11],
			   [0,0,0.11]]
indices=[0, 0]
jointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]

sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius)
ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia]*3, 
							baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex = VisualShapeId, 
							basePosition = [0,0,2])

baseCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.3])

plateCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=boxExtents)
plateId = p.createMultiBody(baseMass=1, baseInertialFramePosition=[0, 0, 0], 
							baseCollisionShapeIndex=baseCollisionShapeId, baseVisualShapeIndex=VisualShapeId, 
							basePosition = [0,0,0.3],
							)

# Read this: http://alexanderfabisch.github.io/pybullet.html

# cubeStartPos = [0,0,1]
# cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)
# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
p.setGravity(0,0,-9.81)

p.setPhysicsEngineParameter(fixedTimeStep=1/1000.)

useRealTimeSimulation = 1

if (useRealTimeSimulation):
	p.setRealTimeSimulation(1)

while 1:
	if (useRealTimeSimulation):
		sleep(1/1000.) # Time in seconds.
	else:
		p.stepSimulation()

	# cp = p.getContactPoints( plateId, ballId )
	# if len(cp) > 0:
		# print(cp[0][5])

	print( p.getBasePositionAndOrientation(plateId) )

	# cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
	# print(cubePos,cubeOrn)
