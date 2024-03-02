#%% 

import numpy as np

def base_side(name, verticle, color):
    size = ".2 0.2 1" if verticle else "0.2 1 0.2"
    pos = .4
    origin = "0 {} {}".format(
        pos if name == "right" else -pos if name == "left" else 0, 
        pos if name == "top" else -pos if name == "bottom" else 0)
    return(
""" 
  <link name="{}">
    <visual>
      <geometry>
        <box size="{}"/>
      </geometry>
      <material name="{}_material">
        <color rgba="{}"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size="{}"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>
  
  <!-- Joints connecting base to the sides -->
  <joint name="{}_joint" type="fixed">
    <parent link="base"/>
    <child link="{}"/>
    <origin xyz="{}" rpy="0 0 0"/>
  </joint>
""".format(name, size, name, color, size, name, name, origin))

base = \
"""
<?xml version="1.0"?>
<robot name="shape">
  <!-- Definition of the invisible base with invisible sides -->

  <link name="base">
    <visual>
      <geometry>
        <box size=".1 .99 .99"/>
      </geometry>
      <material name="base_material">
        <color rgba="0 0 0 0"/> 
      </material>
    </visual>
    <collision>
      <geometry>
          <box size=".1 .99 .99"/>
      </geometry>
    </collision>
    <inertial>
        <mass value="0"/>
        <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>
"""

base += base_side("left", True, "1 0 0 0")
base += base_side("right", True, "0 1 0 0")
base += base_side("bottom", False, "0 0 1 0")
base += base_side("top", False, "1 1 0 0")



square = \
"""
  <!-- Definition of the shape-->
  <link name="square">
    <visual>
      <geometry>
        <box size="0.2 1 1"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <inertial>
        <mass value="0"/>
        <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>

  <joint name="joint5" type="fixed">
    <parent link="base"/>
    <child link="square"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

</robot>
"""

def triangle(name):
    text = "<!-- Definition of the shape-->\n"
    
    def generate_list(start, end, length):
      indices = np.linspace(0, 1, length)
      decay_rate = 1
      values = start + (end - start) * (1 - np.exp(-decay_rate * indices))
      return(values)
    
    length = 5
    widths = generate_list(1, .2, length)
    print(widths)
    for i, width in enumerate(widths):
        text += \
""" 
  <link name="{}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 {} {}"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <inertial>
        <mass value="0"/>
        <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>

  <joint name="{}_joint" type="fixed">
    <parent link="base"/>
    <child link="{}"/>
    <origin xyz="0 {} {}" rpy="0 0 0"/>
  </joint>
""".format(i, 
           width if name in ["up", "down"] else 1/len(widths), 
           width if name in ["left", "right"] else 1/len(widths), 
           i, i, 
           .45 - i/len(widths) if name == "left" else -.45 + i/len(widths) if name == "right" else 0,
           .45 - i/len(widths) if name == "down" else -.45 + i/len(widths) if name == "up" else 0)
    text += "\n</robot>"
    return(text)

up = triangle("up")
down = triangle("down")
left = triangle("left")
right = triangle("right")

for name, shape in [("1_SQUARE", square), ("2_UP", up), ("3_DOWN", down), ("4_LEFT", left), ("5_RIGHT", right)]:
    with open(name + ".urdf", 'w') as file:
        file.write(base + shape)