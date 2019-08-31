'''
file: donkey_env_asyncio.py
author: Leigh Johnson
date: 2019-08-25

Derived from
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
'''

import asyncio
import json
import logging
import os
import random
import time

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess

logging.getLogger(__name__).addHandler(logging.NullHandler())

class AsynMultiDiscreteDonkeyEnv(gym.Env):
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

    async def __init__(self, level, time_step=0.05, frame_skip=2, steer_actions=32, 
            steer_precision=3, throttle_actions=5, throttle_precision=1,
            headless=False,
            port=9090
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

        logging.info("Initializing DonkeyGym env")
        self.port = ports
        self.headless = headless
        self.steer_actions = steer_actions
        self.throttle_actions = throttle_actions

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
        
        # start Unity sim child process
        self.init_donkey_sim()
        await asyncio.sleep(2)

        # init communications
        self.viewer = UnitySimContoller(
            level=level, time_step=time_step, port=self.port
        )

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
        self.view

    def init_donkey_sim(self):
        # start Unity simulation subprocess
        self.proc = DonkeyUnityProcess()

        try:
            exe_path = os.environ['DONKEY_SIM_PATH']
        except:
            logging.error("Missing DONKEY_SIM_PATH environment var. you must start sim manually")
            exe_path = "self_start"
            
        headless = bool(os.environ.get('DONKEY_SIM_HEADLESS', self.headless))
        self.proc.start(exe_path, headless=headless, port=self.port)

class UnitySimController(object):

    '''
        Handles high-level TCP calls between gym.Env and Unity simulator process
    '''
    async def __init__(self, level, time_step=0.05, hostname='0.0.0.0',
                 port=9090, max_cte=5.0, loglevel='INFO', cam_resolution=(120, 160, 3)):

        self.broker = UnitySimBroker(
            level,
            hostname=hostname,
            port=port,
            time_step=time_step, 
            max_cte=max_cte,
            cam_resolution=cam_resolution
        )

    async def wait_until_loaded(self):
        
        while not self.broker.loaded:
            logging.warning("waiting for sim to start..")
            await time.sleep(3.0)

    def reset(self):
        self.broker.reset()

    def get_sensor_size(self):
        return self.broker.get_sensor_size()

    def take_action(self, action):
        self.broker.take_action(action)

    def observe(self):
        return self.broker.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.broker.is_game_over()

    def calc_reward(self, done):
        return self.broker.calc_reward(done)

class UnitySimBroker(object):

    '''Handles TCP connections, messages, and low-level instructions
    Attributes:
    '''
    def __init__(self, 
        level, 
        time_step=0.05, 
        max_cte=5.0, 
        cam_resolution=(120, 160, 3),
        hostname='0.0.0.0',
        port=9090,
        time_step=0.05, 
        max_cte=5.0,
        cam_resolution=None,
        chunk_size=(16 * 1024)
        ):
        '''[summary]
        
        Arguments:
            level {[type]} -- [description]
        
        Keyword Arguments:
            time_step {float} -- [description] (default: {0.05})
            max_cte {float} -- [description] (default: {5.0})
            cam_resolution {[type]} -- [description] (default: {None})
            hostname {str} -- [description] (default: {'0.0.0.0'})
            port {int} -- [description] (default: {9090})
            time_step {float} -- [description] (default: {0.05})
            max_cte {float} -- [description] (default: {5.0})
            cam_resolution {[type]} -- [description] (default: {None})
        '''

        self.hostname = hostname
        self.port = port
        self.chunk_size = chunk_size


        asyncio.start_server(
            self.on_client_connect, 
            host=self.hostname, 
            port=self.port,
            limit=chunk_size
        )


        self.iSceneToLoad = level
        self.time_step = time_step
        self.loaded = False
        self.max_cte = max_cte
        # self.timer = FPSTimer()

        # sensor size - height, width, depth
        self.camera_img_size = cam_resolution


        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False
        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}

    def on_client_connect(self, reader, writer):
        logger.info(f'client connected on port: {self.port}')

        self.reader = reader
        self.writer = writer

    async def handle_read(self):
        data = await reader.read(self.chunk_size)
        chunk = data.decode()
        msg = json.loads(chunk)

        if 'msg_type' not in message:
            logging.error('expected msg_type field')
            return
        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logging.warning(f'unknown message type {msg_type}')

    ## ------- Env interface ---------- ##

    async def reset(self):
        logging.debug("reseting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = self.image_array
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.over = False
        self.send_reset_car()
        # self.timer.reset()
        await asyncio.sleep(1)

    def get_sensor_size(self):
        return self.camera_img_size

    def take_action(self, action):
        self.send_control(action[0], action[1])

    def observe(self):
        while self.last_obs is self.image_array:
            time.sleep(1.0 / 120.0)

        self.last_obs = self.image_array
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {'pos': (self.x, self.y, self.z), 'cte': self.cte,
                "speed": self.speed, "hit": self.hit}

        self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self):
        return self.over

    ## ------ RL interface ----------- ##

    def calc_reward(self, done):
        if done:
            return -1.0

        if self.cte > self.max_cte:
            return -1.0

        if self.hit != "none":
            return -2.0

        # going fast close to the center of lane yeilds best reward
        return 1.0 - (self.cte / self.max_cte) * self.speed

    ## ------ Socket interface ----------- ##

    def on_telemetry(self, data):

        img_str = data['image']
        image = Image.open(BytesIO(base64.b64decode(img_str)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.speed = data["speed"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in data:
            self.cte = data["cte"]

        # don't update hit once session over
        if self.over:
            return

        self.hit = data["hit"]

        self.determine_episode_over()

    def determine_episode_over(self):
        # we have a few initial frames on start that are sometimes very large CTE when it's behind
        # the path just slightly. We ignore those.
        if math.fabs(self.cte) > 2 * self.max_cte:
            pass
        elif math.fabs(self.cte) > self.max_cte:
            logging.debug(f"game over: cte {self.cte}")
            self.over = True
        elif self.hit != "none":
            logging.debug(f"game over: hit {self.hit}")
            self.over = True

    def on_scene_selection_ready(self, data):
        logging.debug("SceneSelectionReady ")
        self.send_get_scene_names()

    def on_car_loaded(self, data):
        logging.debug("car loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        if data:
            names = data['scene_names']
            logging.debug(f"SceneNames: {names}")
            self.send_load_scene(names[self.iSceneToLoad])

    def send_control(self, steer, throttle):
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(
        ), 'throttle': throttle.__str__(), 'brake': '0.0'}
        writer.write(msg.encode())

    def send_reset_car(self):
        msg = {'msg_type': 'reset_car'}
        self.send_control(0, 0)
        writer.write(msg.encode())

    def send_get_scene_names(self):
        msg = {'msg_type': 'get_scene_names'}
        writer.write(msg.encode())

    def send_load_scene(self, scene_name):
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        writer.write(msg.encode())


