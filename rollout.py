"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import utility_funcs
import numpy as np
import os
import sys
import shutil

import argparse
from social_dilemmas.envs.cleanup import CleanupEnv
from social_dilemmas.envs.harvest import HarvestEnv
from social_dilemmas.envs.cartpole import CartPoleEnv


#####################################
parser = argparse.ArgumentParser(description='')

parser.add_argument(
    '--vid_path',
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), './videos')),
    help='Path to directory where videos are saved.')

parser.add_argument(
    '--env',
    type=str,
    default='cleanup',
    help='Name of the environment to rollout.')

parser.add_argument(
    '--render_type',
    type=str,
    default='pretty',
    help='Can be pretty or fast. Implications obvious.')

parser.add_argument(
    '--num_agents',
    type=int,
    default=5,
    help='Number of agents.')

parser.add_argument(
    '--fps',
    type=int,
    default=8,
    help='Number of frames per second.')

args = parser.parse_args()

#######################################



class Controller(object):

    def __init__(self, env_name='cleanup'):
        self.env_name = env_name

        if env_name == 'harvest':
            print('Initializing Harvest environment')
            self.env = HarvestEnv(num_agents=args.num_agents, render=True)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            self.env = CleanupEnv(num_agents=args.num_agents, render=True)
        else:
            print('Error! Not a valid environment type')
            return

        self.env.reset()

        # TODO: initialize agents here






    def rollout(self, horizon=50, save_path=None):
        """ Rollout several timesteps of an episode of the environment.

        Args:
            horizon: The number of timesteps to roll out.
            save_path: If provided, will save each frame to disk at this
                location.
        """
        rewards = []
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]

        for i in range(horizon):
            agents = list(self.env.agents.values())
            action_dim = agents[0].action_space.n
            rand_action = np.random.randint(action_dim, size=args.num_agents)

            obs, rew, dones, info, = self.env.step({('agent-' + str(i)) : rand_action[i]  for i in range(0, args.num_agents)})


            sys.stdout.flush()

            if save_path is not None:
                self.env.render(filename=save_path + 'frame' + str(i).zfill(6) + '.png')

            rgb_arr = self.env.map_to_colors()
            full_obs[i] = rgb_arr.astype(np.uint8)
            observations.append(obs['agent-0'])
            rewards.append(rew['agent-0'])


        return rewards, observations, full_obs

    def render_rollout(self, horizon=50, path=None,
                       render_type='pretty', fps=8):
        """ Render a rollout into a video.

        Args:
            horizon: The number of timesteps to roll out.
            path: Directory where the video will be saved.
            render_type: Can be 'pretty' or 'fast'. Impliciations obvious.
            fps: Integer frames per second.
        """
        if path is None:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            print(path)
            if not os.path.exists(path):
                os.makedirs(path)
        video_name = self.env_name + '_trajectory'

        if render_type == 'pretty':
            image_path = os.path.join(path, 'frames/')
            if not os.path.exists(image_path):
                os.makedirs(image_path)


            rewards, observations, full_obs = self.rollout(horizon=horizon, save_path=image_path)
            utility_funcs.make_video_from_image_dir(path, image_path, fps=fps,
                                                    video_name=video_name)

            # Clean up images
            shutil.rmtree(image_path)
        else:
            rewards, observations, full_obs = self.rollout(horizon=horizon)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path, fps=fps,
                            video_name=video_name)


c = Controller(env_name=args.env)
c.render_rollout(path=args.vid_path, render_type=args.render_type, fps=args.fps)

