'''
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
'''
import os
import random
import time

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess


class MultiDiscreteDonkeyEnv(gym.Env):
    '''
        Multi-discrete action space consists of a series of discrete action spaces, each with a different number of actions

        Used to represent multi-variate agent decisions, for example:
        - A game controller or keyboard where each key is represented by a discrete action space
        - A car controlller with STEER and THROTTLE action spaces

        Reference:
        https://github.com/openai/gym/blob/master/gym/spaces/multi_discrete.py
        e.g. Nintendo Game Controller
        - Can be conceptualized as 3 discrete action spaces:
            1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
            2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
            3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        - Can be initialized as
            MultiDiscrete([ 5, 2, 2 ])

    '''

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 5.0
    VAL_PER_PIXEL = 255

    def __init__(self, level, time_step=0.05, frame_skip=2, steer_actions=32, 
            steer_precision=3, throttle_actions=5, throttle_precision=1,
            headless=False,
            thread_name='train',
            thread_map=None
        ):
        '''
            Donkey Car Unity Sim accepts float32 values for STEER and THROTTLE commands
        
            STEER_RANGE = (STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT)
            THROTTLE_RANGE = (THORTTLE_MIN, THROTTLE_MAX)

            MultiDiscrete environment will divide steering and throttle ranges into discrete actions

            steer_actions - number of discrete steering actions
            throttle_actions - number of discrete throttle actions
            brake_actions - @todo number of discrete brake actions (this should always be 0 or 1?)

            The precision represents the number of significant digits used as a step in the range. 
        '''

        print("starting DonkeyGym env")
        self.headless = headless
        self.steer_actions = steer_actions
        self.throttle_actions = throttle_actions
        self.thread_name = thread_name
        self.thread_map = thread_map

        # Quantize STEER domain
        # Divide range of STEER controls in n bins, where n is `steer_actions`
        self.steer_bins = np.linspace(
            self.STEER_LIMIT_LEFT,
            self.STEER_LIMIT_RIGHT,
            num=steer_actions,
            dtype=np.float32
        )
        
        # Quantize THROTTLE domain
        # Divide range of THROTTLE controls into n bins, where n is `throttle_actions`
        self.throttle_bins = np.linspace(
            self.THROTTLE_MIN,
            self.THROTTLE_MAX,
            num=throttle_actions,
            dtype=np.float32
        )

        self.init_donkey_sim()
        # start simulation com
        time.sleep(2)
        self.viewer = DonkeyUnitySimContoller(
            level=level, time_step=time_step, port=self.port, thread_name=thread_name, thread_map=thread_map)

        # steering and throttle
        self.action_space = spaces.MultiDiscrete([
            steer_actions, throttle_actions
        ])
    
        # camera sensor data
        self.observation_space = spaces.Box(
            0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def init_donkey_sim(self):
        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        try:
            exe_path = os.environ['DONKEY_SIM_PATH']
        except:
            print("Missing DONKEY_SIM_PATH environment var. you must start sim manually")
            exe_path = "self_start"

        try:
            port_offset = 0
            # if more than one sim running on same machine set DONKEY_SIM_MULTI = 1
            random_port = os.environ['DONKEY_SIM_MULTI'] == '1'
            if random_port:
                port_offset = random.randint(0, 1000)
        except:
            pass

        try:
            port = int(os.environ['DONKEY_SIM_PORT']) + port_offset
        except:
            port = 9091 + port_offset
            print("Missing DONKEY_SIM_PORT environment var. Using default:", port)

        headless = bool(os.environ.get('DONKEY_SIM_HEADLESS', self.headless))
        self.port = port
        self.proc.start(exe_path, headless=headless, port=port)

    def __del__(self):
        self.close()

    def close(self):
        self.proc.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def deserialize_action(self, action):
        '''
            action - (steer, throttle) integers representing bin number

            Returns
                (steer_value, throttle_value) - returns quantized float32 values 
        '''

        steer_action, throttle_action = action

        return self.steer_bins[steer_action], self.throttle_bins[throttle_action]

    def step(self, action):

        desearialized_action = self.deserialize_action(action)
        for i in range(self.frame_skip):
            self.viewer.take_action(desearialized_action)
            observation, reward, done, info = self.viewer.observe()
        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        observation, reward, done, info = self.viewer.observe()
        time.sleep(1)
        return observation

    def render(self, mode="human", close=False):
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self):
        return self.viewer.is_game_over()


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    ACTION_NAMES = ["steer", "throttle"]
    STEER_LIMIT_LEFT = -1.0
    STEER_LIMIT_RIGHT = 1.0
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 5.0
    VAL_PER_PIXEL = 255

    def __init__(self, level, time_step=0.05, frame_skip=2):

        print("starting DonkeyGym env")

        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        try:
            exe_path = os.environ['DONKEY_SIM_PATH']
        except:
            print("Missing DONKEY_SIM_PATH environment var. you must start sim manually")
            exe_path = "self_start"

        try:
            port_offset = 0
            # if more than one sim running on same machine set DONKEY_SIM_MULTI = 1
            random_port = os.environ['DONKEY_SIM_MULTI'] == '1'
            if random_port:
                port_offset = random.randint(0, 1000)
        except:
            pass

        try:
            port = int(os.environ['DONKEY_SIM_PORT']) + port_offset
        except:
            port = 9091 + port_offset
            print("Missing DONKEY_SIM_PORT environment var. Using default:", port)

        try:
            headless = os.environ['DONKEY_SIM_HEADLESS'] == '1'
        except:
            print("Missing DONKEY_SIM_HEADLESS environment var. Using defaults")
            headless = False

        self.proc.start(exe_path, headless=headless, port=port)

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(
            level=level, time_step=time_step, port=port)

        # steering and throttle
        self.action_space = spaces.Box(low=np.array([self.STEER_LIMIT_LEFT, self.THROTTLE_MIN]),
                                       high=np.array([self.STEER_LIMIT_RIGHT, self.THROTTLE_MAX]), dtype=np.float32)

        # camera sensor data
        self.observation_space = spaces.Box(
            0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = frame_skip

        # wait until loaded
        self.viewer.wait_until_loaded()

    def __del__(self):
        self.close()

    def close(self):
        self.proc.quit()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        for i in range(self.frame_skip):
            self.viewer.take_action(action)
            observation, reward, done, info = self.viewer.observe()
        return observation, reward, done, info

    def reset(self):
        self.viewer.reset()
        observation, reward, done, info = self.viewer.observe()
        time.sleep(1)
        return observation

    def render(self, mode="human", close=False):
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self):
        return self.viewer.is_game_over()


## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

# Continuous Envs
class GeneratedRoadsEnv(DonkeyEnv):

    def __init__(self):
        super(GeneratedRoadsEnv, self).__init__(level=0)


class WarehouseEnv(DonkeyEnv):

    def __init__(self):
        super(WarehouseEnv, self).__init__(level=1)


class AvcSparkfunEnv(DonkeyEnv):

    def __init__(self):
        super(AvcSparkfunEnv, self).__init__(level=2)


class GeneratedTrackEnv(DonkeyEnv):

    def __init__(self):
        super(GeneratedTrackEnv, self).__init__(level=3)

# Discrete Envs

class MultiDiscreteGeneratedRoadsEnv(MultiDiscreteDonkeyEnv):

    def __init__(self, **kwargs):
        super(MultiDiscreteGeneratedRoadsEnv, self).__init__(level=0, **kwargs)


class MultiDiscreteWarehouseEnv(MultiDiscreteDonkeyEnv):

    def __init__(self, **kwargs):
        super(MultiDiscreteWarehouseEnv, self).__init__(level=1, **kwargs)


class MultiDiscreteAvcSparkfunEnv(MultiDiscreteDonkeyEnv):

    def __init__(self, **kwargs):
        super(MultiDiscreteAvcSparkfunEnv, self).__init__(level=2, **kwargs)


class MultiDiscreteGeneratedTrackEnv(MultiDiscreteDonkeyEnv):

    def __init__(self, **kwargs):
        super(MultiDiscreteGeneratedTrackEnv, self).__init__(level=3, **kwargs)
