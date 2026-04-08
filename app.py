import gradio as gr
import threading
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction
import random, requests, os

# ─── FastAPI (for hackathon checker) ───
api = FastAPI()
env = EmergencyResourceEnv()
obs = env.reset()

@api.post("/reset")
def reset():
    global obs
    obs = env.reset()
    return JSONResponse(obs.to_dict())

@api.post("/step")
def step(data: dict):
    global obs
    action = EmergencyAction.from_int(int(data.get("action", 0)))
    result = env.step(action)
    obs, reward, done, _ = result.as_tuple()
    return JSONResponse({
        "observation": obs.to_dict(),
        "reward": reward,
        "done": done
    })

@api.get("/state")
def state():
    return JSONResponse(obs.to_dict())

# Start FastAPI in background
def run_api():
    uvicorn.run(api, host="0.0.0.0", port=7861)

threading.Thread(target=run_api, daemon=True).start()

# ─── Gradio (your dashboard) ───
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    return response.json()

def run():
    global obs
    obs = env.reset()
    last_action = -1
    total_reward = 0
    output_text = "🚨 Emergency Resource Allocation AI\n"
    output_text += "=" * 40 + "\n\n"

    for step in range(21):
        state_text = str(obs.to_dict())
        prompt = f"""You are an ambulance controller.
State: {state_text}
There are 8 actions (0-7).
Previous action: {last_action}
Rules: Do NOT repeat previous action.
Return ONLY a number (0-7)."""
        try:
            result = query(prompt)
            action = int(result[0]["generated_text"].strip()[0])
            if action == last_action:
                action = random.randint(0, 7)
        except:
            action = random.randint(0, 7)

        last_action = action
        action_obj = EmergencyAction.from_int(action)
        result = env.step(action_obj)
        obs, reward, done, _ = result.as_tuple()
        total_reward += reward

        output_text += f"📍 Step {step+1}\n"
        output_text += f"   🚑 Action: {action_obj}\n"
        output_text += f"   ⭐ Reward: {reward:.2f}\n"
        output_text += "-" * 30 + "\n"
        if done:
            break

    output_text += f"\n✅ Done! Total Reward: {total_reward:.2f}\n"
    return output_text

demo = gr.Interface(
    fn=run,
    inputs=[],
    outputs="text",
    title="🚨 Emergency Resource Allocation AI",
    description="AI agent allocates ambulances using RL."
)

demo.launch()
