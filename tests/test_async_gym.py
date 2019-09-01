

import pytest

import gym
import gym_donkeycar.envs
async_env_list = [
       "donkey-warehouse-async-multidiscrete-v0",
       "donkey-generated-roads-async-multidiscrete-v0",
       "donkey-avc-sparkfun-async-multidiscrete-v0",
       "donkey-generated-track-async-multidiscrete-v0"
]

def test_load_async_gyms(mocker):
    sim_ctl = mocker.patch('gym_donkeycar.envs.donkey_env_async.UnitySimController')
    unity_proc = mocker.patch('gym_donkeycar.envs.donkey_env_async.DonkeyUnityProcess')

    for i, gym_name in enumerate(async_env_list):

        env = gym.make(gym_name, steer_actions=32, throttle_actions=5)
        assert env.ACTION_NAMES == ['steer', 'throttle']
        assert env.spec.id == gym_name
        assert sim_ctl.call_count == i+1
        assert unity_proc.call_count == i+1
        
        assert len(env.steer_bins) == 32
        assert len(env.throttle_bins) == 5

        #import pdb; pdb.set_trace()

        assert env.deserialize_action((0, 0)) == (env.STEER_LIMIT_LEFT, env.THROTTLE_MIN)
        assert env.deserialize_action((31, 0)) == (env.STEER_LIMIT_RIGHT, env.THROTTLE_MIN)
        assert env.deserialize_action((0, 4)) == (env.STEER_LIMIT_LEFT, env.THROTTLE_MAX)
        assert env.deserialize_action((31, 4)) == (env.STEER_LIMIT_RIGHT, env.THROTTLE_MAX)
