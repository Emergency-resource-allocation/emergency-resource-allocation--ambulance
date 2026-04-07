from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction
import random

def run_simulation():
    env = EmergencyResourceEnv()
    obs = env.reset()

    total_reward = 0

    print("🚑 Simulation Started\n")

    for step in range(20):
        action = EmergencyAction.from_int(random.randint(0, 7))
        result = env.step(action)

        total_reward += result.reward

        print(f"Step {step+1}")
        print("Reward:", result.reward)
        print("Done:", result.done)
        print("-" * 30)

        if result.done:
            break

    print("\n🏁 Final Reward:", total_reward)


if __name__ == "__main__":
    run_simulation()