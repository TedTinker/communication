try:
    from .part import Part  
except ImportError:
    from part import Part  

arm_thickness = .5

parts = [
    Part(
        name = "body", 
        mass = 100, 
        size = (1, 1, 1),
        sensors = 1,
        sensor_sides = ["start", "stop", "top", "left", "right"]),
    
    Part(
        name = "back_left_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (-.35, .5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed"),
    
    Part(
        name = "from_left_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (.35, .5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed"),
    
    Part(
        name = "back_right_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (-.35, -.5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed"),
    
    Part(
        name = "from_right_wheel", 
        mass = .1, 
        shape = "cylinder",
        size = (.1, .1, .1), 
        joint_parent = "body", 
        joint_origin = (.35, -.5, -.4), 
        joint_axis = (1, 0, 0),
        joint_rpy=(1.5708, 0, 0),
        joint_type = "fixed"),
    
    Part(
        name = "joint_1", 
        mass = .1, 
        size = (.4, .3, arm_thickness), 
        joint_parent = "body", 
        joint_origin = (0, .65, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous"),
    
    Part(
        name = "left_arm", 
        mass = .1, 
        size = (3, .4, arm_thickness), 
        joint_parent = "joint_1", 
        joint_origin = (1.3, .35, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "left_hand",
        mass = .1,
        size = (.4, .7, arm_thickness),
        joint_parent = "left_arm", 
        joint_origin = (1.3, -.55, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "stop"]),
        
    Part(
        name = "joint_2",
        mass = .1,
        size = (.4, .3, arm_thickness),
        joint_parent = "body", 
        joint_origin = (0, -.65, 0), 
        joint_axis = (0, -1, 0),
        joint_type = "continuous"),
    
    Part(
        name = "right_arm", 
        mass = .1, 
        size = (3, .4, arm_thickness), 
        joint_parent = "joint_2", 
        joint_origin = (1.3, -.35, 0), 
        joint_axis = (1, 0, 0),
        joint_type = "fixed",
        sensors = 3,
        sensor_angle = 0),
    
    Part(
        name = "right_hand",
        mass = .1,
        size = (.4, .7, arm_thickness),
        joint_parent = "right_arm", 
        joint_origin = (1.3, .55, 0), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_angle = 1,
        sensor_sides = ["top", "bottom", "left", "right", "start"])
]