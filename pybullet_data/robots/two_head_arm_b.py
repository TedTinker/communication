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
hand_width = 1

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
        name = "back_left_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (-.35, .5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
    
    Part(
        name = "from_left_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (.35, .5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
    
    Part(
        name = "back_right_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (-.35, -.5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
    
    Part(
        name = "from_right_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (.35, -.5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed",
        inertia = [.0005, 0, 0, .0005, 0, .0005]),
        
    Part(
        name = "joint_1",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, joint_1_height),
        joint_parent = "body", 
        joint_origin = (0, 0, .5 + joint_1_height / 2), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        sensors = 1,
        sensor_sides = ["left", "right", "start", "stop"],
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
        sensors = 1,
        sensor_sides = ["left", "right", "top", "stop"],
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
        sensor_sides = ["left", "right", "top", "bottom", "start"],
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