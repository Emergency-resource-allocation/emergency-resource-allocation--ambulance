from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from emergency_env.environment import EmergencyResourceEnv
from emergency_env.models import EmergencyAction
import uvicorn, random, requests, os

app = FastAPI()
env = EmergencyResourceEnv()
obs = env.reset()

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.post("/reset")
def reset():
    global obs
    obs = env.reset()
    return JSONResponse(obs.to_dict())

@app.post("/step")
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

@app.get("/state")
def state():
    return JSONResponse(obs.to_dict())

@app.get("/")
def home():
    return HTMLResponse("""
    <html>
    <head>
        <title>Emergency Resource Allocation AI</title>
        <style>
            body { font-family: Arial; background: #0a0a0f; color: white; 
                   text-align: center; padding: 50px; }
            h1 { color: #ff6b6b; }
            button { background: #ff6b6b; color: white; border: none;
                     padding: 15px 40px; font-size: 18px; 
                     border-radius: 8px; cursor: pointer; margin: 10px; }
            #output { background: #1a1a2e; padding: 20px; border-radius: 10px;
                      text-align: left; margin-top: 20px; 
                      min-height: 200px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h1>🚨 Emergency Resource Allocation AI</h1>
        <p>AI agent allocates ambulances using Reinforcement Learning</p>
        <button onclick="runSim()">▶ Run Simulation</button>
        <button onclick="resetSim()">🔄 Reset</button>
        <div id="output">Click Run Simulation to start...</div>
        <script>
        async function resetSim() {
            let r = await fetch('/reset', {method:'POST'});
            let d = await r.json();
            document.getElementById('output').innerText = 
                '✅ Environment Reset!\\n' + JSON.stringify(d, null, 2);
        }
        async function runSim() {
            document.getElementById('output').innerText = '⏳ Running...';
            await fetch('/reset', {method:'POST'});
            let log = '🚨 Emergency Resource Allocation AI\\n';
            log += '='.repeat(40) + '\\n\\n';
            let totalReward = 0;
            for(let i = 0; i < 10; i++) {
                let action = Math.floor(Math.random() * 8);
                let r = await fetch('/step', {
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body: JSON.stringify({action: action})
                });
                let d = await r.json();
                totalReward += d.reward;
                log += `📍 Step ${i+1}\\n`;
                log += `   🚑 Action: ${action}\\n`;
                log += `   ⭐ Reward: ${d.reward.toFixed(2)}\\n`;
                log += '-'.repeat(30) + '\\n';
                document.getElementById('output').innerText = log;
                if(d.done) break;
            }
            log += `\\n✅ Done! Total Reward: ${totalReward.toFixed(2)}`;
            document.getElementById('output').innerText = log;
        }
        </script>
    </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
