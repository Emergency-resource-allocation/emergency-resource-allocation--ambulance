import numpy as np
from src.environment import EmergencyEnv

class GreedyAgent:
    """Always assigns the closest available ambulance to the patient who has waited longest."""
    def act(self, state: dict) -> tuple:
        available_ambs = np.where(state["available_mask"])[0]
        active_pats = np.where(state["active_mask"])[0]
        
        if len(available_ambs) == 0 or len(active_pats) == 0:
            return None
        
        # Priority: Patient waiting the longest
        target_pat = active_pats[np.argmax(state["wait_times"][active_pats])]
        
        # Find closest ambulance to that patient
        pat_loc = state["patients"][target_pat]
        amb_locs = state["ambulances"][available_ambs]
        distances = np.linalg.norm(amb_locs - pat_loc, axis=1)
        best_amb = available_ambs[np.argmin(distances)]
        
        return (best_amb, target_pat)

def run_simulation(difficulty="Medium"):
    env = EmergencyEnv(difficulty=difficulty)
    agent = GreedyAgent()
    state = env.reset()
    
    print(f"\n--- Starting Simulation: {difficulty} ---")
    total_reward = 0
    done = False
    
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total_reward += reward
        
    # Grading: Normalize based on patient count
    max_possible = env.n_patients * 100
    score = max(0, min(1.0, total_reward / max_possible))
    
    print(f"Simulation Ended. Total Reward: {total_reward:.2f}")
    print(f"Final Normalized Score: {score:.4f}")
    return score

if __name__ == "__main__":
    run_simulation("Hard") # 15 ambulances vs 40 patients