import math

try:
    from .part import Part  
except ImportError:
    from part import Part  
    
arm_mass = 2
arm_thickness = .2
arm_length = 1.75

joint_1_height = .2

hand_length = 1
hand_height = 1.25
hand_width = arm_thickness * 3


parts = [
    
    Part(
        name = "body",
        mass = 100,
        size = (1, 1, 1),
        joint_origin = (0, 0, 1.05), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        inertia = [15, 0, 0, 15, 0, 15]),
    
    Part(
        name = "left_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.2, .1, .1), 
        joint_parent = "body", 
        joint_origin = (0, .5, -.3), 
        joint_axis = (0, 0, 1),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "continuous",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
    
    Part(
        name = "left_spoke_1",
        mass = 0,
        size = (.3, .03, .1),
        joint_parent = "left_wheel", 
        joint_origin = (0, 0, -.01), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = "left_spoke_2",
        mass = 0,
        size = (.03, .3, .1),
        joint_parent = "left_wheel", 
        joint_origin = (0, 0, -.01), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = "right_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.2, .1, .1), 
        joint_parent = "body", 
        joint_origin = (0, -.5, -.3), 
        joint_axis = (0, 0, 1),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "continuous",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
    
    Part(
        name = "right_spoke_1",
        mass = 0,
        size = (.3, .03, .1),
        joint_parent = "right_wheel", 
        joint_origin = (0, 0, .01), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = "right_spoke_2",
        mass = 0,
        size = (.03, .3, .1),
        joint_parent = "right_wheel", 
        joint_origin = (0, 0, .01), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        inertia = [0, 0, 0, 0, 0, 0]),
    
    Part(
        name = "joint_1",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, joint_1_height),
        joint_parent = "body", 
        joint_origin = (0, 0, .5 + joint_1_height / 2), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        joint_limits = [-math.pi/4, math.pi/4, 999, 999],
        inertia = [.01, 0, 0, .01, 0, .01]),
    
    Part(
        name = "joint_2",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, arm_thickness),
        joint_parent = "joint_1", 
        joint_origin = (0, 0, joint_1_height / 2 + arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "continuous",
        joint_limits = [-math.pi/2, 0, 999, 999],
        inertia = [.01, 0, 0, .01, 0, .01]),
    
    Part(
        name = "arm",
        mass = arm_mass,
        size = (arm_length, arm_thickness, arm_thickness),
        joint_parent = "joint_2", 
        joint_origin = (arm_thickness / 2 + arm_length / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["left", "right", "top", "bottom"],
        inertia = [0.01, 0, 0, 1.25, 0, 1.25]),
    
    Part(
        name = "hand",
        mass = arm_mass,
        size = (hand_length, hand_width, hand_height),
        joint_parent = "arm", 
        joint_origin = (arm_length / 2 + hand_length / 2, 0, - hand_height / 2 + arm_thickness / 2),
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1),
    
    ]