#%% 

import matplotlib.pyplot as plt
import pybullet as p
from time import sleep
from math import pi

from utils import args, shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration, wait_for_button_press, plot_number_bars #, print
from arena import Arena, get_physics

sleep_time = .5
verbose = True
waiting = True
sleep_time = 1

x = (args.max_object_distance**2 / 2) ** .5



task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
    num_objects = 1,
    allowed_tasks_and_weights = [(0, 1)],
    allowed_colors = [0],
    allowed_shapes = [2],
    test = None)

do_these = [
    "show_movements",
    #"watch",
    "push",
    "pull",
    "right",
    "left",
    ]


    
def execute_task(task_name, task_id, moves, set_positions):
    print(f"\n\n###\n### {task_name.upper()}!\n###\n\n")
    goal = Goal(task_map[task_id], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting=True)
    arena.begin(objects=colors_shapes_1, goal=goal, parenting=False, set_positions=[set_positions])
    show_them()
    arena.rewards(verbose=True)
    
    for lw, rw, j1, j2 in moves:
        arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
        plot_number_bars([lw, rw, j1, j2])
        show_them()
        reward, win, mother_voice = arena.rewards(verbose=True)
        if win:
            break
            
    wait_for_button_press()
    arena.end()

def show_them():
    fontsize = 10
    """above_rgba = arena.photo_from_above()
    plt.imshow(above_rgba)
    plt.show()
    plt.close()"""
    agent_rgbd = arena.photo_for_agent()
    plt.figure(figsize=(4, 4))
    plt.imshow(agent_rgbd[:,:,:-1])
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.show()
    plt.close()



args.robot_name = "two_head_arm"
args.min_joint_1_angle = -pi/4
args.max_joint_1_angle = pi/4
args.min_joint_2_angle = -pi/2
args.max_joint_2_angle = 0
args.consideration = 1



physicsClient = get_physics(GUI = True, args = args)
arena = Arena(physicsClient, args = args)



if(args.robot_name == "one_head_arm"):
    
    if("show_movements" in do_these):
        moves = [[1, -1, -1, None], [-1, 1, 1, None]] 
        execute_task(
            task_name = "show_movements", 
            task_id = 0, 
            moves = moves, 
            set_positions = (9, 0))

    if("watch" in do_these):
        moves = [[-.5, .5, -1, None]] * 1 + [[0, 0, -1, None]] * 10
        execute_task(
            task_name = "watch", 
            task_id = 1, 
            moves = moves, 
            set_positions = (x, x))
            
    if("push" in do_these):
        moves =  [[-.5, .5, -1, None]] * 1 + [[1, 1, -1, None]] * 10
        execute_task(
            task_name = "push", 
            task_id = 2, 
            moves = moves, 
            set_positions = (x, x))

    if("pull" in do_these):
        moves = [[-.5, .5, -1, None]] * 1 + [[1, 1, -.25, None]] * 1 + [[0, 0, .7, None]] + [[-1, -1, 0, None]] * 10
        execute_task(
            task_name = "pull", 
            task_id = 3, 
            moves = moves, 
            set_positions = (x, x))
    
    if("left" in do_these):
        moves = [[-.5, .5, -.3, None]] * 1 + [[0, 0, .3, None]] * 5
        execute_task(
            task_name = "left", 
            task_id = 4, 
            moves = moves, 
            set_positions = (x, x))
        
  
    if("right" in do_these):
        moves = [[.5, -.5, .3, None]] * 1 + [[0, 0, -.3, None]] * 5
        execute_task(
            task_name = "right", 
            task_id = 5, 
            moves = moves, 
            set_positions = (x, -x))
        


if(args.robot_name == "two_head_arm"):
    if("show_movements" in do_these):
        moves = [[1, -1, 1, 1], [-1, 1, -1, -1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        execute_task(
            task_name = "show_movements", 
            task_id = 0, 
            moves = moves, 
            set_positions = (9, 0))
        
    if("watch" in do_these):
        moves = [[-.5, .5, 0, 0]] * 1 + [[0, 0, 0, 0]] * 10
        execute_task(
            task_name = "watch", 
            task_id = 1, 
            moves = moves, 
            set_positions = (x, x))

    if("push" in do_these):
        moves =  [[-.5, .5, 0, 0]] * 1 + [[1, 1, 0, 0]] * 10
        execute_task(
            task_name = "push", 
            task_id = 2, 
            moves = moves, 
            set_positions = (x, x))

    if("pull" in do_these):
        moves = [[-.5, .5, 0, 0]] * 1 + [[1, 1, 0, 0]] * 1 + [[0, 0, 0, 1]] + [[-1, -1, 0, 0]] * 10
        execute_task(
            task_name = "pull", 
            task_id = 3, 
            moves = moves, 
            set_positions = (x, x))
        
    if("left" in do_these):
        moves = [[-.5, .5, -.3, 1]] * 1 + [[0, 0, .3, 1]] * 5
        execute_task(
            task_name = "left", 
            task_id = 4, 
            moves = moves, 
            set_positions = (x, x))
        
    if("right" in do_these):
        moves = [[.5, -.5, .3, 1]] * 1 + [[0, 0, -.3, 1]] * 5
        execute_task(
            task_name = "right", 
            task_id = 5, 
            moves = moves, 
            set_positions = (x, -x))
        
        
        
if(args.robot_name == "two_head_arm_b"):
    
    if("show_movements" in do_these):
        moves = [[1, -1, 1, 1], [-1, 1, -1, -1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        execute_task(
            task_name = "show_movements", 
            task_id = 0, 
            moves = moves, 
            set_positions = (9, 0))
        
    if("watch" in do_these):
        moves = [[-.5, .5, 0, 0]] * 1 + [[0, 0, 0, 0]] * 10
        execute_task(
            task_name = "watch", 
            task_id = 1, 
            moves = moves, 
            set_positions = (x, x))

    if("push" in do_these):
        moves =  [[-.5, .5, 0, 0]] * 1 + [[1, 1, 0, 0]] * 10
        execute_task(
            task_name = "push", 
            task_id = 2, 
            moves = moves, 
            set_positions = (x, x))

    if("pull" in do_these):
        moves = [[-.5, .5, 0, 0]] * 1 + [[0, 0, 0, 1]] * 1 + [[-1, -1, 0, 1]] * 10
        execute_task(
            task_name = "pull", 
            task_id = 3, 
            moves = moves, 
            set_positions = (x, x))
        
    if("left" in do_these):
        moves = [[-.5, .5, -.3, 1]] * 1 + [[0, 0, .3, 1]] * 5
        execute_task(
            task_name = "left", 
            task_id = 4, 
            moves = moves, 
            set_positions = (x, x))
        
    if("right" in do_these):
        moves = [[.5, -.5, .3, 1]] * 1 + [[0, 0, -.3, 1]] * 5
        execute_task(
            task_name = "right", 
            task_id = 5, 
            moves = moves, 
            set_positions = (x, -x))
        
        
        
        
if(args.robot_name == "two_head_arm_c"):
    if("show_movements" in do_these):
        moves = [[1, -1, 1, 1], [-1, 1, -1, -1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        execute_task(
            task_name = "show_movements", 
            task_id = 0, 
            moves = moves, 
            set_positions = (9, 0))
        
        
        
if(args.robot_name == "two_head_arm_d"):
    if("show_movements" in do_these):
        moves = [[1, -1, 1, 1], [-1, 1, -1, -1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        execute_task(
            task_name = "show_movements", 
            task_id = 0, 
            moves = moves, 
            set_positions = (9, 0))
            



        
goal = Goal(task_map[0], task_map[0],task_map[0], parenting = True)
arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(10, 10)])
while(True):
    p.stepSimulation(physicsClientId = arena.physicsClient)
    sleep(.0001)
# %%
