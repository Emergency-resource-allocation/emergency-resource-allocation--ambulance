from emergency_env.environment import EmergencyResourceEnv

# Initialize environment
env = EmergencyResourceEnv()

def reset():
    obs = env.reset()
    return obs.to_dict()

def step(action):
    result = env.step(action)
    obs, reward, done, info = result.as_tuple()

    return {
        "observation": obs.to_dict(),
        "reward": reward,
        "done": done,
        "info": info
    }
