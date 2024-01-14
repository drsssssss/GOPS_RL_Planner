import numpy as np
from matplotlib.axes import Axes


def plot_vx_vy_r(eval_dict, fig, gs):
    # ego vx, vy, yaw rate
    ax1_1: Axes = fig.add_subplot(gs[0, 1])
    ax1_1.plot(eval_dict['step_list'], eval_dict['vx_list'], '.-', label='vx', color='b')
    ax1_1.set_ylabel('$v_x$', color='b')
    ax1_1.tick_params('y', colors='b')
    ax1_2 = ax1_1.twinx()
    ax1_2.plot(eval_dict['step_list'], eval_dict['vy_list'], '.-', label='vy', color='r')
    ax1_2.set_ylabel('$v_y$', color='r')
    ax1_2.tick_params('y', colors='r')
    ax1_3 = ax1_1.twinx()
    ax1_3.spines['right'].set_position(
        ('outward', 60))  # Move the last y-axis spine over to the right by 60 points
    ax1_3.plot(eval_dict['step_list'], eval_dict['r_list'], '.-', label='yaw rate', color='g')
    ax1_3.set_ylabel('yaw rate', color='g')
    ax1_3.tick_params('y', colors='g')
    ax1_1.set_title('Vehicle Speeds and Yaw Rate')
    ax1_1.set_xlabel('Time')
    return ax1_1, ax1_2, ax1_3


def plot_y_ref_phi_ref(eval_dict, fig, gs):
    # reference delta y and delta phi
    ax2_1: Axes = fig.add_subplot(gs[1, 1])
    ax2_1.plot(eval_dict['step_list'], eval_dict['y_ref_list'], '.-', label='lateral error')
    ax2_1.set_ylabel('$y-y_{ref}$', color='b')
    ax2_1.tick_params('y', colors='b')
    ax2_2 = ax2_1.twinx()
    # ax2_2.plot(eval_dict['step_list'], eval_dict['phi_ref_list'], '.-',
    #            label='relative orientation', color='r')
    ax2_2.plot(eval_dict['step_list'], eval_dict['ref_phi_list'], '.-', color='r')
    ax2_2.set_ylabel('$phi_{ref}$ (degree)', color='r')
    ax2_2.tick_params('y', colors='r')
    ax2_2.set_title('Errors with Reference Trajectory')
    ax2_3 = ax2_1.twinx()
    ax2_3.spines['right'].set_position(
        ('outward', 60))  # Move the last y-axis spine over to the right by 60 points
    ax2_3.plot(eval_dict['step_list'], eval_dict['ego_phi_list'], '.-', color='g')
    ax2_3.set_ylabel('$phi$ (degree)', color='g')
    ax2_3.tick_params('y', colors='g')

    ax2_4 = ax2_1.twinx()
    ax2_4.spines['right'].set_position(
        ('outward', 120))  # Move the last y-axis spine over to the right by 60 points
    ax2_4.plot(eval_dict['step_list'], eval_dict['rel_ego_y_list'], '.-', color='c')
    ax2_4.set_ylabel('$rel y$ (m)', color='c')
    ax2_4.tick_params('y', colors='c')

    ax2_1.set_xlabel('Time')
    return ax2_1, ax2_2, ax2_3, ax2_4


def plot_reward(episode_data, eval_dict, fig, gs):
    # all the non-zero reward
    ax3_1: Axes = fig.add_subplot(gs[2, 1])
    for k, v in episode_data.reward_info.items():
        if np.abs(np.mean(v)) > 1e-6:
            ax3_1.plot(eval_dict['step_list'], v, '.-', label=k)
    ax3_1.set_ylabel('Reward')
    ax3_1.set_title('Rewards')
    # ax3_1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax3_1.set_xlabel('Time')
    return ax3_1


def plot_action(eval_dict, fig, gs):
    # real acc, real steer angle
    ax4_1: Axes = fig.add_subplot(gs[3, 1])
    ax4_1.plot(eval_dict['step_list'], eval_dict['acc_list'], '.-', label='acceleration',
               color='b')
    ax4_1.set_ylabel('acceleration', color='b')
    ax4_1.tick_params('y', colors='b')
    ax4_2 = ax4_1.twinx()
    ax4_2.plot(eval_dict['step_list'], eval_dict['steer_list'], '.-', label='steering angle',
               color='r')
    ax4_2.set_ylabel('steering angle (degree)', color='r')
    ax4_2.tick_params('y', colors='r')
    ax4_1.set_xlabel('Time')
    ax4_1.set_title('Actual Acceleration and Steering Angle')
    return ax4_1, ax4_2
