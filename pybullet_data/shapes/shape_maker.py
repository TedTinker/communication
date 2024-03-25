#%% 

import numpy as np

max_radius = .6

base = \
"""
<?xml version="1.0"?>
<robot name="shape">
  <!-- Definition of the base -->

  <link name="base">
    <visual>
      <geometry>
        <cylinder length=".1" radius="{}"/>
      </geometry>
      <material name="base_material">
        <color rgba="1 1 1 1"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <cylinder length=".1" radius="{}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="10"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>
""".format(max_radius, max_radius)



def innards(lengths, radia):
    text = "<!-- Definition of the shape -->\n"
    
    length_so_far = .05
    for i, (length, radius) in enumerate(zip(lengths, radia)):
        text += \
""" 
  <link name="{}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="{}" radius="{}"/>
      </geometry>
      <material name="red">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
          <cylinder length="{}" radius="{}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value=".1"/>
        <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
    </inertial>
  </link>

  <joint name="{}_joint" type="fixed">
    <parent link="base"/>
    <child link="{}"/>
    <origin xyz="0 0 {}" rpy="0 0 0"/>
  </joint>
""".format(i, 
           length,
           radius, 
           length,
           radius, 
           i, i, 
           length/2 + length_so_far)
        length_so_far += length
    text += "\n</robot>"
    return(text)

pole = innards([1], [max_radius])
bottom = innards([1], [.1])
both = innards([.9, .1], [.1, max_radius])
middle = innards([.45, .1, .45], [.1, max_radius, .1])
delta = innards([.1] * 10, [max_radius - i/16 for i in range(10)])

for name, shape in [("0_L_POLE", pole), ("1_M_BOTTOM", bottom), ("2_N_BOTH", both), ("3_O_MIDDLE", middle), ("4_P_DELTA", delta)]:
    with open(name + ".urdf", 'w') as file:
        file.write(base + shape)