from gym.envs.registration import register

register(
    id='ccl-env-cpd-v1',
    entry_point='gym_cell_model.envs:CCLEnvCPD'
)