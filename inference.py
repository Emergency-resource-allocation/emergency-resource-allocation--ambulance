from fastapi import FastAPI
from emergency_env.environment import EmergencyResourceEnv

app = FastAPI()

# Initialize the environment globally
env = EmergencyResourceEnv()

@app.post("/reset")
async def reset_endpoint():
    try:
        obs = env.reset()
        # Ensure obs.to_dict() is returning a standard Python dict
        return obs.to_dict() 
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
async def step_endpoint(action: dict):
    # 'action' comes from the POST body
    result = env.step(action['action']) 
    obs, reward, done, info = result.as_tuple()
    
    return {
        "observation": obs.to_dict(),
        "reward": reward,
        "done": done,
        "info": info
    }
