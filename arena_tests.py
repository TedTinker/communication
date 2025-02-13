#%% 

import matplotlib.pyplot as plt
import pybullet as p
from time import sleep
from math import pi

from utils import args, default_args, shape_map, color_map, task_map, Goal, empty_goal, relative_to, opposite_relative_to, make_objects_and_task, duration, wait_for_button_press, plot_number_bars #, print
from arena import Arena, get_physics

args = default_args

sleep_time = .5
verbose = True
waiting = True
sleep_time = 1


    
# THIS SHOWS THE ADD_STEPS HYPERPARAMS AREN'T RIGHT!
x = 1
"""
args.max_steps = int(10 * x)
args.time_step = .2 / x
args.steps_per_step = int(20 / x)
args.push_amount = .75 / x
args.pull_amount = .25 / x
args.left_right_amount = .25 / x"""



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

task, colors_shapes_1, colors_shapes_2 = make_objects_and_task(
    num_objects = 1,
    allowed_tasks_and_weights = [(0, 1)],
    allowed_colors = [0],
    allowed_shapes = [2],
    test = None)

do_these = [
    #"show_movements",
    #"watch",
    #"push",
    #"pull",
    "right",
    "left",
    ]


if(args.robot_name == "one_head_arm"):
    
    physicsClient = get_physics(GUI = True, args = args)
    arena = Arena(physicsClient, args = args)

    if("show_movements" in do_these):
        print("\nSHOW MOVEMENT")
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(9,0)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[1, -1, -1], [-1, 1, 1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()



    if("watch" in do_these):
        print("\nWATCH")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[1], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, x)])
        show_them()
        reward, win, mother_voice = arena.rewards(verbose = True)
        moves = [[-.5, .5, -1]] * 1 + [[0, 0, -1]] * 10
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
            
    if("push" in do_these):
        print("\nPUSH")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x,x)])
        show_them()
        arena.rewards(verbose = True)
        moves =  [[-.5, .5, -1]] * 1 + [[1, 1, -1]] * 10
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()

    if("pull" in do_these):
        print("\nPULL") 
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x,x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[-.5, .5, -1]] * 1 + [[1, 1, -1]] * 1 + [[0, 0, 1]] + [[-1, -1, 0]] * 10
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()

    if("right" in do_these):
        print("\nRIGHT")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[5], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, -x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[.4, -.4, 0]] * 1 + [[0, 0, -.3]] * 5
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
        
    if("left" in do_these):
        print("\nLEFT")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[-.4, .4, 0]] * 1 + [[0, 0, .3]] * 5
        for lw, rw, j1 in moves:
            arena.step(lw, rw, j1, 0, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
        




if(args.robot_name == "two_head_arm"):
    
    args.min_joint_2_angle = -pi/2
    args.max_joint_2_angle = 0

    physicsClient = get_physics(GUI = True, args = args)
    arena = Arena(physicsClient, args = args)
    
    if("show_movements" in do_these):
        print("\nSHOW MOVEMENT")
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(9,0)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[1, -1, 1, 1], [-1, 1, -1, -1]] # [[1, -1, 1]] * 4 + [[-1, 1, -1]] * 4
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
        
        
        



    if("watch" in do_these):
        print("\nWATCH")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[1], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, x)])
        show_them()
        reward, win, mother_voice = arena.rewards(verbose = True)
        moves = [[-.5, .5, 0, 0]] * 1 + [[0, 0, 0, 0]] * 10
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
            
    if("push" in do_these):
        print("\nPUSH")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[2], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x,x)])
        show_them()
        arena.rewards(verbose = True)
        moves =  [[-.5, .5, 0, 0]] * 1 + [[1, 1, 0, 0]] * 10
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()

    if("pull" in do_these):
        print("\nPULL") 
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[3], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x,x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[-.5, .5, 0, 0]] * 1 + [[1, 1, 0, 0]] * 1 + [[0, 0, 0, 1]] + [[-1, -1, 0, 0]] * 10
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()

    if("right" in do_these):
        print("\nRIGHT")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[5], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, -x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[.4, -.4, 0, 1]] * 1 + [[0, 0, -.3, 0]] * 5
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()
        
    if("left" in do_these):
        print("\nLEFT")
        x = (args.max_object_distance**2 / 2) ** .5
        goal = Goal(task_map[4], colors_shapes_1[0][0], colors_shapes_1[0][1], parenting = True)
        arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(x, x)])
        show_them()
        arena.rewards(verbose = True)
        moves = [[-.4, .4, 0, 1]] * 1 + [[0, 0, .3, 0]] * 5
        for lw, rw, j1, j2 in moves:
            arena.step(lw, rw, j1, j2, verbose = verbose, sleep_time = sleep_time, waiting = waiting)
            plot_number_bars([lw, rw, j1, j2])
            show_them()
            reward, win, mother_voice = arena.rewards(verbose = True)
            if(win):
                break
        wait_for_button_press()
        arena.end()


        
goal = Goal(task_map[0], task_map[0],task_map[0], parenting = True)
arena.begin(objects = colors_shapes_1, goal = goal, parenting = False, set_positions = [(10, 10)])
while(True):
    p.stepSimulation(physicsClientId = arena.physicsClient)
    sleep(.0001)