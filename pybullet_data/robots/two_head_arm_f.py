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
        name = "wrist",
        mass = arm_mass,
        size = (arm_thickness, arm_thickness, wrist_length),
        joint_parent = "arm", 
        joint_origin = (arm_length / 2 - arm_thickness / 2, 0, - wrist_length / 2 - arm_thickness / 2),
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["start", "stop", "left", "right"]),
    
    Part(
        name = "hand_part_0",
        mass = arm_mass,
        size = (arm_thickness, hand_part_length, hand_part_height),
        joint_parent = "wrist", 
        joint_origin = (0, 0, - wrist_length / 2 - hand_part_height / 2),
        joint_axis = (0, 1, 0),
        joint_type = "fixed",
        sensors = 1,
        sensor_sides = ["bottom", "top", "start", "stop"]),
    

    
    ]

# Apothem (distance from circle center to chord midpoint)
a = hand_part_length / (2 * math.tan(math.pi / number_of_hand_parts))

for hand_part_number in range(1, number_of_hand_parts):
    
    if(hand_part_number <= ignore_these_hand_parts or hand_part_number >= number_of_hand_parts - ignore_these_hand_parts):

        theta = -math.pi/2 + (2 * math.pi * hand_part_number / number_of_hand_parts)
        center_x = a * (1 - math.cos(2 * math.pi * hand_part_number / number_of_hand_parts))
        center_y = a * math.sin(2 * math.pi * hand_part_number / number_of_hand_parts)

        parts.append(
                Part(
                    name = f"hand_part_{hand_part_number}",
                    mass = arm_mass,
                    size = (hand_part_length, arm_thickness, hand_part_height),
                    joint_parent = f"hand_part_0", 
                    joint_origin = (-center_x, center_y, 0),   
                    joint_axis = (0, 1, 0),
                    joint_type = "fixed",
                    sensors = 1,
                    sensor_sides = ["bottom", "top", "left", "right"],
                    joint_rpy=(0, 0, theta)),   
        )
        