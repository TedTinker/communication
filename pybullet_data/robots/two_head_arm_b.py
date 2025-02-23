try:
    from .part import Part  
except ImportError:
    from part import Part  
    
arm_thickness = .4
arm_length = 2.75
hand_length_1 = 1.3
hand_length_2 = 1
arm_mass = 2

parts = [
    
    Part(
        name = "body",
        mass = 100,
        size = (1, 1, 1),
        joint_origin = (0, 0, 1.05), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1),
        
    Part(
        name = "joint_1",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, .2),
        joint_parent = "body", 
        joint_origin = (0, 0, .5 + .1), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"]),
    
    Part(
        name = "joint_2",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, .4),
        joint_parent = "joint_1", 
        joint_origin = (0, 0, .3), 
        joint_axis = (0, 1, 0),
        joint_type = "continuous",
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"]),
    
    Part(
        name = "arm",
        mass = arm_mass,
        size = (arm_length, arm_thickness, arm_thickness),
        joint_parent = "joint_2", 
        joint_origin = (arm_thickness / 2 + arm_length / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "bottom", "top", "left", "right"]),
    
    Part(
        name = "hand_1",
        mass = arm_mass,
        size = (arm_thickness, hand_length_1, 1),
        joint_parent = "arm", 
        joint_origin = (arm_length / 2 - arm_thickness / 2, 0, -1/2 - arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "bottom", "left", "right"]),
    
    Part(
        name = "hand_2",
        mass = arm_mass,
        size = (hand_length_2, arm_thickness, 1),
        joint_parent = "hand_1", 
        joint_origin = (arm_thickness / 2 - hand_length_2 / 2, arm_thickness / 2 + hand_length_1 / 2, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "bottom", "left", "right"]),
    
    Part(
        name = "hand_3",
        mass = arm_mass,
        size = (hand_length_2, arm_thickness, 1),
        joint_parent = "hand_1", 
        joint_origin = (arm_thickness / 2 - hand_length_2 / 2, -arm_thickness / 2 - hand_length_1 / 2, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "bottom", "left", "right"]),
    
    ]