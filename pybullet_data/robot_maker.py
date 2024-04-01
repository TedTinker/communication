#%% 

import numpy as np



sensor_width = .02



robot = \
"""<?xml version="1.0"?>
<robot name="robot">"""



def make_part(name, mass, shape, with_sensor = False):
    global robot
    robot += \
f"""\n\n
    <!-- {name} -->
    <link name="{name}">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="{mass}"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0]} {shape[1]} {shape[2]}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0]} {shape[1]} {shape[2]}"/>
            </geometry>
        </collision>
    </link>"""
    
    if(with_sensor):
        robot += \
f"""\n\n
    <!-- {name}_sensor -->
    <link name="{name}_sensor">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0"/>
            <inertia ixx=".1" ixy=".1" ixz=".1" iyy=".1" iyz=".1" izz=".1"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0] + sensor_width} {shape[1] + sensor_width} {shape[2] + sensor_width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{shape[0] + sensor_width} {shape[1] + sensor_width} {shape[2] + sensor_width}"/>
            </geometry>
        </collision>
    </link>
    
    
	<!-- Joint: {name}, {name}_sensor -->
    <joint name="{name}_{name}_sensor_joint" type="fixed"> 
        <parent link="{name}"/> 
        <child link="{name}_sensor"/> 
        <origin xyz="0 0 0" rpy="0 0 0"/> 
        <axis xyz="0 0 0"/> 
    </joint>"""
    
    
    
def make_joint(parent, child, origin, axis, type = "fixed"):
    global robot
    robot += \
f"""\n\n
    <!-- Joint: {parent}, {child} -->
    <joint name="{parent}_{child}_joint" type="{type}"> 
        <parent link="{parent}"/> 
        <child link="{child}"/> 
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/> 
        <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/> 
    </joint>"""



make_part(  
          name = "body",          
          mass = 100,     
          shape = (1, 1, 1),          
          with_sensor = True)
make_part(  
          name = "nose",          
          mass = 0,       
          shape = (.001, .2, .2),     
          with_sensor = False)
make_joint( 
           parent = "body",        
           child = "nose", 
           origin = (.50005, 0, 0),    
           axis = (0, 1, 0),   
           type = "fixed")

make_part(  
          name = "left_shoulder", 
          mass = .1,       
          shape = (.1, .5, .8),       
          with_sensor = False)
make_joint( 
           parent = "body",        
           child = "left_shoulder", 
           origin = (0, .75, 0),    
           axis = (0, -1, 0),   
           type = "continuous")

make_part(  
          name = "left_arm", 
          mass = .1,       
          shape = (3, .1, .8),       
          with_sensor = True)
make_joint( 
           parent = "left_shoulder",        
           child = "left_arm", 
           origin = (1.45, .25, 0),    
           axis = (1, 0, 0),   
           type = "fixed")

make_part(  
          name = "hand", 
          mass = .1,       
          shape = (.1, 2, .8),       
          with_sensor = True)
make_joint( 
           parent = "left_arm",        
           child = "hand", 
           origin = (1.5, -1, 0),    
           axis = (0, 0, 0),   
           type = "fixed")

make_part(  
          name = "right_arm", 
          mass = .1,       
          shape = (3, .1, .8),       
          with_sensor = True)
make_joint( 
           parent = "hand",        
           child = "right_arm", 
           origin = (-1.5, -1, 0),    
           axis = (1, 0, 0),   
           type = "fixed")

make_part(  
          name = "right_shoulder", 
          mass = .1,       
          shape = (.1, .5, .8),       
          with_sensor = False)
make_joint( 
           parent = "right_arm",        
           child = "right_shoulder", 
           origin = (-1.45, .25, 0),    
           axis = (0, -1, 0),   
           type = "fixed")



robot += \
"""\n\n</robot>"""

with open("robot.urdf", 'w') as file:
    file.write(robot)
# %%
