import math

try:
    from .part import Part  
except ImportError:
    from part import Part  
    
arm_mass = 2
arm_thickness = .2
arm_length = 2.75

joint_1_height = .2

wrist_length = .1

number_of_hand_parts = 16
ignore_these_hand_parts = 5
hand_radius = 1
hand_part_height = 1
hand_part_length = 2 * hand_radius * math.sin(math.pi / number_of_hand_parts)
hand_angle = .5

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
        name = "head_part",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, joint_1_height),
        joint_parent = "body", 
        joint_origin = (0, 0, .5 + joint_1_height / 2), 
        joint_axis = (0, 0, 1),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["left", "right", "start", "stop"],
        joint_limits = [-math.pi/4, math.pi/4, 999, 999],
        inertia = [.01, 0, 0, .01, 0, .01]),
    
    Part(
        name = "joint_1",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, arm_thickness),
        joint_parent = "head_part", 
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
        joint_parent = "joint_1", 
        joint_origin = (arm_thickness / 2 + arm_length / 2, 0, 0), 
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["left", "right", "top", "bottom", "start"],
        inertia = [0.01, 0, 0, 1.25, 0, 1.25]),
    
    Part(
        name = "wrist",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, wrist_length),
        joint_parent = "arm", 
        joint_origin = (arm_length / 2 - arm_thickness / 2, 0, - wrist_length / 2 - arm_thickness / 2),
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "left", "right"],
        inertia = [0.01, 0, 0, 0.01, 0, 0.01]),
    
    ]

# Apothem (distance from circle center to chord midpoint)
a = hand_part_length / (2 * math.tan(math.pi / number_of_hand_parts))

for hand_part_number in range(0, number_of_hand_parts):
    
    making_hand_part = hand_part_number <= ignore_these_hand_parts or hand_part_number >= number_of_hand_parts - ignore_these_hand_parts
    first_hand_part = hand_part_number == ignore_these_hand_parts
    last_hand_part = hand_part_number == number_of_hand_parts - ignore_these_hand_parts
    
    if(hand_part_number <= ignore_these_hand_parts or hand_part_number >= number_of_hand_parts - ignore_these_hand_parts):

        theta = -math.pi/2 + (2 * math.pi * hand_part_number / number_of_hand_parts)
        center_x = a * (1 - math.cos(2 * math.pi * hand_part_number / number_of_hand_parts))
        center_y = a * math.sin(2 * math.pi * hand_part_number / number_of_hand_parts)

        sensor_sides = ["bottom", "top", "left", "right"] + (["stop"] if first_hand_part else ["start"] if last_hand_part else [])

        parts.append(
                Part(
                    name = f"hand_part_{hand_part_number}",
                    mass = arm_mass,
                    size = (hand_part_length, arm_thickness, hand_part_height),
                    joint_parent = f"wrist", 
                    joint_origin = (-center_x + arm_thickness, center_y, -hand_part_height / 2),   
                    joint_axis = (0, 1, 0),
                    joint_type = "fixed",
                    sensors = 1,
                    sensor_sides = sensor_sides,
                    joint_rpy=(hand_angle, 0, theta),
                    inertia = [0.2, 0, 0, 0.2, 0, 0.03]),   
        )
        