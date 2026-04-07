from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction
import random

def evaluate():
    env = EmergencyResourceEnv()
    obs = env.reset()

    total_reward = 0

    for _ in range(20):
        action = EmergencyAction.from_int(random.randint(0,7))
        result = env.step(action)
        total_reward += result.reward

        if result.done:
            break

    return total_reward


if __name__ == "__main__":
    print("Score:", evaluate())