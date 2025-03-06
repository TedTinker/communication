#%%

import math

try:
    from .part import Part  
except ImportError:
    from part import Part  
    


def compute_offsets(L1, L2, theta1, theta2):
    offset_x = (L1 / 2.0) * math.cos(theta1) + (L2 / 2.0) * math.cos(theta2)
    offset_z = (L1 / 2.0) * math.sin(theta1) + (L2 / 2.0) * math.sin(theta2)
    return offset_x, offset_z


    
arm_mass = 2
arm_thickness = .2
arm_length = 2.75

joint_1_height = .2

triangle_radius = 1
triangle_center = arm_length / 2 - triangle_radius
palm_center_offset_x = triangle_radius / 4
palm_center_offset_y = palm_center_offset_x * (3**.5)



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
        name="palm_1",
        mass=arm_mass,
        size=(arm_thickness, triangle_radius, arm_thickness),
        joint_parent="arm",
        joint_origin=(triangle_center - palm_center_offset_x, -palm_center_offset_y, 0),  
        joint_rpy=(0, 0, -math.radians(30)),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    
    Part(
        name="palm_2",
        mass=arm_mass,
        size=(arm_thickness, triangle_radius, arm_thickness),
        joint_parent="arm",
        joint_origin=(triangle_center - palm_center_offset_x, palm_center_offset_y, 0),  
        joint_rpy=(0, 0, math.radians(30)),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    

]



finger_3 = [] 



current_angle = 90

finger_angle_1 = 70
finger_length_1 = 1.5

finger_offset_1_x, finger_offset_1_y = compute_offsets(
    arm_length, 
    finger_length_1,
    math.radians(current_angle),
    math.radians(finger_angle_1)
)



finger_3.append(
    Part(
        name="finger_1_3",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, finger_length_1),
        joint_parent="arm",
        joint_origin=(finger_offset_1_y, 0, -finger_offset_1_x),  
        joint_rpy=(0, -math.radians(finger_angle_1), 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    )
)



# Not working after first finger.

finger_angle_2 = 25
finger_length_2 = .5

finger_offset_2_x, finger_offset_2_y = compute_offsets(
    finger_length_1, 
    finger_length_2,
    math.radians(finger_angle_1),
    math.radians(finger_angle_2)
)



finger_3.append(
    Part(
        name="finger_2_3",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, finger_length_2),
        joint_parent="finger_1_3",
        joint_origin=(finger_offset_2_y, 0, -finger_offset_2_x),  
        joint_rpy=(0, -math.radians(finger_angle_2), 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    )
)


 
parts += finger_3













    
    
'''    
[
    
    Part(
        name="hand_1_a",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, hand_length_1),
        joint_parent="palm_1",
        joint_origin=(0, -triangle_radius / 2 + arm_thickness / 2, -hand_length_1 / 2 - arm_thickness / 2),  
        joint_rpy=(0, 0, 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    
    Part(
        name="hand_2_a",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, hand_length_1),
        joint_parent="palm_2",
        joint_origin=(0, triangle_radius / 2 - arm_thickness / 2, -hand_length_1 / 2 - arm_thickness / 2),  
        joint_rpy=(0, 0, 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),

    Part(
        name="hand_3_a",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, hand_length_1),
        joint_parent="arm",
        joint_origin=(arm_length / 2 - arm_thickness / 2 , 0, -hand_length_1 / 2 - arm_thickness / 2),  
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    
    
    
    
    
    Part(
        name="hand_3_b",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, hand_length_2),
        joint_parent="hand_3_a",
        joint_origin=(hand_length_2 * finger_offset_x, 0, -hand_length_1 / 2 - hand_length_2 * finger_offset_y),  
        joint_rpy=(0, -math.radians(finger_angle), 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    
    
    
    
    
    
    
    
    
    
    
    
    Part(
        name="hand_3_c",
        mass=arm_mass,
        size=(arm_thickness, arm_thickness, hand_length_2),
        joint_parent="hand_3_b",
        joint_origin=(hand_length_3 * finger_offset_x, 0, -hand_length_1 / 2 - hand_length_3 * finger_offset_y),  
        joint_rpy=(0, math.radians(finger_angle), 0),
        joint_axis=(0, 1, 0),
        joint_type="fixed",
        sensors=1
    ),
    


]'''