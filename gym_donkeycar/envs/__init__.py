# flake8: noqa
from gym.envs.registration import register
from .donkey_env import GeneratedRoadsEnv, WarehouseEnv, AvcSparkfunEnv, GeneratedTrackEnv

# Continuous Envs
register(
    id='donkey-generated-roads-v0',
    entry_point='gym_donkeycar.envs.donkey_env:GeneratedRoadsEnv',
)

register(
    id='donkey-generated-track-v0',
    entry_point='gym_donkeycar.envs.donkey_env:GeneratedTrackEnv',
)

register(
    id='donkey-warehouse-v0',
    entry_point='gym_donkeycar.envs.donkey_env:WarehouseEnv',
)

register(
    id='donkey-avc-sparkfun-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AvcSparkfunEnv',
)

# Discrete Envs

register(
    id='donkey-generated-roads-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedRoadsEnv',
)

register(
    id='donkey-generated-track-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteGeneratedTrackEnv',
)

register(
    id='donkey-warehouse-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteWarehouseEnv',
)

register(
    id='donkey-avc-sparkfun-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:MultiDiscreteAvcSparkfunEnv',
)

# Asyncio Envs

register(
    id='donkey-generated-roads-async-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AsyncMultiDiscreteGeneratedRoadsEnv',
)

register(
    id='donkey-generated-track-async-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AsyncMultiDiscreteGeneratedTrackEnv',
)

register(
    id='donkey-warehouse-async-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AsyncMultiDiscreteWarehouseEnv',
)

register(
    id='donkey-avc-sparkfun-async-multidiscrete-v0',
    entry_point='gym_donkeycar.envs.donkey_env:AsyncMultiDiscreteAvcSparkfunEnv',
)