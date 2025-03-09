import math

try:
    from .part import Part  
except ImportError:
    from part import Part  

arm_thickness = .5
arm_len = 3 - arm_thickness

pitch_joint_len = .2
hand_len = .65

parts = [
    Part(
        name = "body", 
        mass = 100, 
        size = (1, 1, 1),
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"],
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
        mass = .1, 
        size = (arm_thickness, pitch_joint_len, arm_thickness), 
        joint_parent = "body", 
        joint_origin = (.5 - arm_thickness/2, .5 + pitch_joint_len/2, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous",
        joint_limits = [0, math.pi/2, 999, 999]),
    
    Part(
        name = "joint_2", 
        mass = .1, 
        size = (arm_thickness, arm_thickness, arm_thickness), 
        joint_parent = "joint_1", 
        joint_origin = (0, (pitch_joint_len + arm_thickness) / 2, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        joint_limits = [-math.pi/4, math.pi/4, 999, 999]),
    
    Part(
        name = "left_arm", 
        mass = .1, 
        size = (arm_len, arm_thickness, arm_thickness), 
        joint_parent = "joint_2", 
        joint_origin = (arm_thickness/2 + arm_len/2, 0, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "left_hand",
        mass = .1,
        size = (arm_thickness, hand_len, arm_thickness),
        joint_parent = "left_arm", 
        joint_origin = (arm_len/2 - arm_thickness/2, - (arm_thickness/2 + hand_len/2), 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "stop"]),
        
        
        
        
        
    Part(
        name = "joint_3", 
        mass = .1, 
        size = (arm_thickness, pitch_joint_len, arm_thickness), 
        joint_parent = "body", 
        joint_origin = (.5 - arm_thickness/2, -(.5 + pitch_joint_len/2), 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous",
        joint_limits = [0, math.pi/2, 999, 999]),
    
    Part(
        name = "joint_4", 
        mass = .1, 
        size = (arm_thickness, arm_thickness, arm_thickness), 
        joint_parent = "joint_3", 
        joint_origin = (0, -(pitch_joint_len + arm_thickness) / 2, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        joint_limits = [-math.pi/4, math.pi/4, 999, 999]),
    
    Part(
        name = "right_arm", 
        mass = .1, 
        size = (arm_len, arm_thickness, arm_thickness), 
        joint_parent = "joint_4", 
        joint_origin = (arm_thickness/2 + arm_len/2, 0, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "right_hand",
        mass = .1,
        size = (arm_thickness, hand_len, arm_thickness),
        joint_parent = "right_arm", 
        joint_origin = (arm_len/2 - arm_thickness/2, (arm_thickness/2 + hand_len/2), 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "start"]),
]