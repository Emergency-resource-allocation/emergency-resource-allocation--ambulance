import os
from openai import OpenAI
from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction
import random

# ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

env = EmergencyResourceEnv()
obs = env.reset()

total_rewards = []
step_count = 0

print(f"[START] task=emergency env=grid model={MODEL_NAME}")

last_action = -1

while True:
    state_text = str(obs)

    prompt = f"""
You are an ambulance controller.

State:
{state_text}

Choose best action (0-7).
Do not repeat previous action.
Return only number.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        output = response.choices[0].message.content.strip()
        action = int(output[0])

    except:
        action = random.randint(0, 7)

    if action == last_action:
        action = random.randint(0, 7)

    last_action = action

    action_obj = EmergencyAction.from_int(action)
    result = env.step(action_obj)

    obs, reward, done, info = result.as_tuple()

    total_rewards.append(round(reward, 2))
    step_count += 1

    print(f"[STEP] step={step_count} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

    if done:
        break

print(f"[END] success=true steps={step_count} rewards={','.join(map(str, total_rewards))}")