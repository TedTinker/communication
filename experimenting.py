#%%
import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

startPos = [0, 0, 0.5]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("robot.urdf", startPos, startOrientation, useFixedBase=False)

num_joints = p.getNumJoints(robotId)
print("Number of joints in the robot:", num_joints)
movable_joints = []
for i in range(num_joints):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]
    if joint_type != p.JOINT_FIXED:
        movable_joints.append(i)
        print(f"Movable joint {i}: {joint_name}")

if len(movable_joints) < 2:
    print("Not enough movable joints found!")
    exit()

joint1 = movable_joints[0]
joint2 = movable_joints[1]

# Set target positions (in radians) for the joints
target_position_joint1 = 0.5   # Example target for the first joint
target_position_joint2 = -0.5  # Example target for the second joint

# Command the joints to move using POSITION_CONTROL
p.setJointMotorControl2(robotId, joint1, controlMode=p.POSITION_CONTROL, 
                        targetPosition=target_position_joint1, force=500)
p.setJointMotorControl2(robotId, joint2, controlMode=p.POSITION_CONTROL, 
                        targetPosition=target_position_joint2, force=500)

# Run the simulation for 5 seconds (assuming simulation runs at 240 Hz)
simulation_duration = 5  # seconds
num_steps = int(240 * simulation_duration)
for step in range(num_steps):
    p.stepSimulation()
    time.sleep(1/240.)

# Disconnect from the simulation
p.disconnect()
