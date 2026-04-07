import requests
import os
import random
from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction

# 👉 YOUR HF TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()

# Initialize environment
env = EmergencyResourceEnv()
obs = env.reset()

total_reward = 0
last_action = -1

print("🤖 Final Smart HF Agent Started\n")

for step in range(21):

    state_text = str(obs.to_dict())

    prompt = f"""
You are an ambulance controller.

State:
{state_text}

There are 8 actions (0-7).

Goal:
- Reach emergency requests quickly
- Reduce delay
- Avoid wasting steps

Previous action: {last_action}

Rules:
- Do NOT repeat previous action
- Choose a DIFFERENT and better action

Return ONLY a number (0-7).
"""

    result = query(prompt)

    try:
        output = result[0]["generated_text"]
        print("RAW:", output)
        action = int(output.strip()[0])

        if action == last_action:
            action = random.randint(0, 7)

    except:
        action = random.randint(0, 7)

    last_action = action

    action_obj = EmergencyAction.from_int(action)
    result = env.step(action_obj)
    obs, reward, done, _ = result.as_tuple()

    total_reward += reward

    print(f"Step {step+1}")
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print("----------------------")

print("🏁 Final Reward:", total_reward)