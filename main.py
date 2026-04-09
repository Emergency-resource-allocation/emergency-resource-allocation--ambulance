import numpy as np
from environment import EmergencyEnv # Note: Import updated for flat structure

class GreedyAgent:
    def act(self, state: dict) -> tuple:
        available_ambs = np.where(state["available_mask"])[0]
        active_pats = np.where(state["active_mask"])[0]
        if len(available_ambs) == 0 or len(active_pats) == 0:
            return None
        target_pat = active_pats[np.argmax(state["wait_times"][active_pats])]
        pat_loc = state["patients"][target_pat]
        amb_locs = state["ambulances"][available_ambs]
        distances = np.linalg.norm(amb_locs - pat_loc, axis=1)
        best_amb = available_ambs[np.argmin(distances)]
        return (best_amb, target_pat)

def run_simulation(difficulty="Medium"):
    env = EmergencyEnv(difficulty=difficulty)
    agent = GreedyAgent()
    state = env.reset()
    total_reward = 0
    done = False
    print(f"\n>>> Running {difficulty} Simulation...")
    
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward
        
    score = max(0, min(1.0, total_reward / (env.n_patients * 100)))
    print(f"Result: Reward = {total_reward:.2f} | Score = {score:.4f}")

if __name__ == "__main__":
    run_simulation("Medium")