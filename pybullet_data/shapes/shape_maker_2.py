#%% 

import os
import numpy as np
import pybullet as p
from math import pi, sin, cos, radians, tan, sqrt
import matplotlib.pyplot as plt
from skimage.transform import resize
from copy import deepcopy

base_size = 1
diamond_size = sqrt(2 * ((base_size/2) ** 2))
thinkness = .01

base = \
f"""
<?xml version="1.0"?>
<robot name="shape">
  <!-- Definition of the base -->

  <link name="base">
    <visual>
      <geometry>
        <box size="{base_size} {base_size} {base_size}"/>
      </geometry>
      <material name="base_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{base_size} {base_size} {base_size}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="100"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
"""

square = \
f""" 
  <link name="square">
    <visual>
      <geometry>
        <box size="{thinkness} {base_size} {base_size}"/>
      </geometry>
      <material name="square_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {base_size} {base_size}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="square_joint" type="fixed">
    <parent link="base"/>
    <child link="square"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy="0 0 0"/>
  </joint>
  
</robot>
"""

cross = \
f""" 
  <link name="cross_1">
    <visual>
      <geometry>
        <box size="{thinkness} {.15} {base_size}"/>
      </geometry>
      <material name="cross_1_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {.15} {base_size}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="cross_1_joint" type="fixed">
    <parent link="base"/>
    <child link="cross_1"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy="0 0 0"/>
  </joint>
  
  <link name="cross_2">
    <visual>
      <geometry>
        <box size="{thinkness} {base_size} {.15}"/>
      </geometry>
      <material name="cross_2_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {base_size} {.15}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="cross_2_joint" type="fixed">
    <parent link="base"/>
    <child link="cross_2"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy="0 0 0"/>
  </joint>
  
</robot>
"""

x = \
f""" 
  <link name="x_1">
    <visual>
      <geometry>
        <box size="{thinkness} {.1} {base_size + .1}"/>
      </geometry>
      <material name="x_1_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {.1} {base_size + .1}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="x_1_joint" type="fixed">
    <parent link="base"/>
    <child link="x_1"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy=".785 0 0"/>
  </joint>
  
  <link name="x_2">
    <visual>
      <geometry>
        <box size="{thinkness} {base_size + .1} {.15}"/>
      </geometry>
      <material name="x_2_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {base_size + .1} {.15}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="x_2_joint" type="fixed">
    <parent link="base"/>
    <child link="x_2"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy=".785 0 0"/>
  </joint>
  
</robot>
"""

diamond = \
f""" 
  <link name="diamond">
    <visual>
      <geometry>
        <box size="{thinkness} {diamond_size} {diamond_size}"/>
      </geometry>
      <material name="diamond_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {diamond_size} {diamond_size}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="diamond_joint" type="fixed">
    <parent link="base"/>
    <child link="diamond"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy=".785 0 0"/>
  </joint>
  
</robot>
"""

hole = \
f""" 
  <link name="square">
    <visual>
      <geometry>
        <box size="{thinkness} {base_size} {base_size}"/>
      </geometry>
      <material name="square_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness} {base_size} {base_size}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="square_joint" type="fixed">
    <parent link="base"/>
    <child link="square"/>
    <origin xyz="{base_size/2 + thinkness/2} 0 0" rpy="0 0 0"/>
  </joint>
  
  <link name="base_hole">
    <visual>
      <geometry>
        <box size="{thinkness*2} {base_size/2} {base_size/2}"/>
      </geometry>
      <material name="base_hole_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{thinkness*2} {base_size/2} {base_size/2}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
  
  <joint name="base_hole_joint" type="fixed">
    <parent link="base"/>
    <child link="base_hole"/>
    <origin xyz="{base_size/2 + thinkness} 0 0" rpy="0 0 0"/>
  </joint>
  
</robot>
"""


current_dir = os.getcwd()
last_folder = os.path.basename(os.getcwd())
if last_folder == "pybullet_data":
  new_dir = os.path.join(current_dir, "shapes")
  os.chdir(new_dir)
  
shapes = [square, x, cross, diamond, hole]
names = ["SQUARE", "CROSS", "X", "DIAMOND", "HOLE"]
letters = ["Q", "R", "S", "T", "U"]
file_names = []

for i in range(len(shapes)):
  file_name = f"{i+5}_{letters[i]}_{names[i]}.urdf"
  shape = shapes[i]
  file_names.append(file_name)
  with open(file_name, 'w') as file:
    file.write(base + shape)
        


physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath("pybullet_data")

these_colors = [
  (1, 0, 0, 1),
  (0, 1, 0, 1),
  (0, 0, 1, 1),
  (0, 1, 1, 1),
  (1, 0, 1, 1)
]


print(file_names)



for index, (i, file_name) in enumerate(zip([-2, -1, 0, 1, 2], file_names)):
  object_index = p.loadURDF("{}".format(file_name), (-5, 15 * i, base_size), p.getQuaternionFromEuler([0, 0, 0]), 
                                              useFixedBase=False, globalScaling = 2, physicsClientId=physicsClient)
  color = these_colors[index]
  p.changeVisualShape(object_index, -1, rgbaColor = (1, 1, 1, 1), physicsClientId = physicsClient)
  for i in range(p.getNumJoints(object_index)):
    joint_info = p.getJointInfo(object_index, i, physicsClientId=physicsClient)
    joint_name = joint_info[1].decode("utf-8")  # Joint name is in byte format, so decode it to a string
    if "base" not in joint_name.lower():  # Check if "base" is not in the joint name (case insensitive)
        p.changeVisualShape(object_index, i, rgbaColor=color, physicsClientId=physicsClient)

  
  
  
fov_x_deg = 90
fov_y_deg = 90
fov_x_rad = radians(fov_x_deg)
fov_y_rad = radians(fov_y_deg)
near = .1
far = 15
right = near * tan(fov_x_rad / 2)
left = -right
top = near * tan(fov_y_rad / 2)
bottom = -top



def photo(pos, image_size):
    x, y = cos(pi), sin(pi)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[pos[0], pos[1], 1], 
        cameraTargetPosition=[pos[0] + x*2, pos[1] + y*2, 1],
        cameraUpVector=[0, 0, 1], physicsClientId=physicsClient)
    proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
    _, _, rgba, depth, _ = p.getCameraImage(
        width=image_size * 2, height=image_size * 2,
        projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow=0,
        physicsClientId=physicsClient)
    rgb = np.divide(rgba[:, :, :-1], 255)
    rgb = resize(rgb, (image_size, image_size, 3))
    return rgb
  
def plot_these(all_rgbs):
  fig, axs = plt.subplots(len(all_rgbs[0]), len(all_rgbs), figsize = (20, 20))
  for i, rgbs in enumerate(all_rgbs):
    for j, rgb in enumerate(rgbs):
      axs[j, i].imshow(rgb)
  fig.show()
  #plt.close()
  

for image_size in [8, 16, 32]:
  all_rgbs = []
  for object_pos in [15 * i for i in [-2, -1, 0, 1, 2]]:
    rgbs = []
    for distance_pos in [p - 5 for p in [2.5, 3, 4, 5, 6, 7]]: 
      rgb = photo((distance_pos, object_pos), image_size)
      rgbs.append(rgb)
    all_rgbs.append(rgbs)
  plot_these(all_rgbs)
  
#p.disconnect(physicsClientId = physicsClient)



"""physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath("pybullet_data")

for i, file_name in zip([-2, -1, 0, 1, 2], file_names):
  object_index = p.loadURDF("{}".format(file_name), (-5, 3 * i, 0), p.getQuaternionFromEuler([0, 0, pi/2]), 
                                              useFixedBase=False, globalScaling = 2, physicsClientId=physicsClient)
  p.changeVisualShape(object_index, -1, rgbaColor = (1, 0, 0, 1), physicsClientId = physicsClient)
  for i in range(p.getNumJoints(object_index)):
      p.changeVisualShape(object_index, i, rgbaColor=(1, 0, 0, 1), physicsClientId = physicsClient)
      
x, y = cos(pi), sin(pi)
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[0, 0, 1], 
    cameraTargetPosition= [x*2, y*2, 1],
    cameraUpVector=[0, 0, 1], physicsClientId=physicsClient)
proj_matrix = p.computeProjectionMatrix(left, right, bottom, top, near, far)
_, _, rgba, depth, _ = p.getCameraImage(
    width = 3600, height = 3600,
    projectionMatrix=proj_matrix, viewMatrix=view_matrix, shadow=0,
    physicsClientId=physicsClient)
rgb = np.divide(rgba[:, :, :-1], 255)

plt.imshow(rgb)
plt.show()

p.disconnect(physicsClientId = physicsClient)"""

  