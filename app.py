import gradio as gr
from main import run_simulation

def run():
    steps_log = []
    total_reward = 0

    from emergency_env.environment import EmergencyResourceEnv
    from emergency_env.models import EmergencyAction
    import random, requests, os

    HF_TOKEN = os.getenv("HF_TOKEN")
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def query(prompt):
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        return response.json()

    env = EmergencyResourceEnv()
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
Goal: Reach emergency requests quickly, reduce delay, avoid wasting steps.
Previous action: {last_action}
Rules: Do NOT repeat previous action. Choose a DIFFERENT and better action.
Return ONLY a number (0-7)."""

        try:
            result = query(prompt)
            action_raw = result[0]["generated_text"]
            action = int(action_raw.strip()[0])
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
        output_text += f"   🚑 Action taken: {action_obj}\n"
        output_text += f"   ⭐ Reward: {reward:.2f}\n"
        output_text += f"   📊 State: {str(obs.to_dict())[:80]}...\n"
        output_text += "-" * 40 + "\n"

        if done:
            break

    output_text += f"\n✅ Simulation Complete!\n"
    output_text += f"🏆 Total Reward: {total_reward:.2f}\n"
    output_text += f"📈 Steps completed: {step+1}/21\n"

    return output_text

demo = gr.Interface(
    fn=run,
    inputs=[],
    outputs="text",
    title="🚨 Emergency Resource Allocation AI",
    description="AI agent allocates ambulances using Reinforcement Learning. Click Generate to run!"
)

demo.launch()
