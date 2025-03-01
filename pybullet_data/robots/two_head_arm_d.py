try:
    from .part import Part  
except ImportError:
    from part import Part  
    
arm_mass = 2
arm_thickness = .2
arm_length = 2.75

joint_1_height = .2

palm_length_1 = 1.5
palm_length_2 = 2.5

hand_length = 1.2

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
        size = (arm_thickness, arm_thickness, joint_1_height),
        joint_parent = "body", 
        joint_origin = (0, 0, .5 + joint_1_height / 2), 
        joint_axis = (0, 0, 1),
        joint_type = "continuous",
        sensors = 1,
        sensor_sides = ["left", "right", "start", "stop"]),
    
    Part(
        name = "joint_2",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, arm_thickness),
        joint_parent = "joint_1", 
        joint_origin = (0, 0, joint_1_height / 2 + arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "continuous",
        sensors = 1,
        sensor_sides = ["left", "right", "top", "stop"]),
    
    Part(
        name = "arm",
        mass = arm_mass,
        size = (arm_length, arm_thickness, arm_thickness),
        joint_parent = "joint_2", 
        joint_origin = (arm_thickness / 2 + arm_length / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["left", "right", "top", "bottom", "start"]),
    
    Part(
        name = "palm",
        mass = arm_mass,
        size = (arm_thickness, palm_length_1, arm_thickness),
        joint_parent = "arm", 
        joint_origin = (arm_length / 2 - palm_length_2 / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1),
    
    Part(
        name = "hand_1",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, hand_length),
        joint_parent = "palm", 
        joint_origin = (palm_length_2 / 2 - arm_thickness / 2, 0, - hand_length / 2 - arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "left", "right", "bottom"]),
    
    Part(
        name = "hand_2",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, hand_length),
        joint_parent = "palm", 
        joint_origin = (0, palm_length_1 / 2 - arm_thickness / 2, - hand_length / 2 - arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "left", "right", "bottom"]),
    
    Part(
        name = "hand_3",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, hand_length),
        joint_parent = "palm", 
        joint_origin = (0, - palm_length_1 / 2 + arm_thickness / 2, - hand_length / 2 - arm_thickness / 2), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "left", "right", "bottom"]),
    
    ]