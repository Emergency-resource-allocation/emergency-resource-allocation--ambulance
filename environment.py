import numpy as np
from typing import Dict, Tuple, Optional

class EmergencyEnv:
    def __init__(self, difficulty: str = 'Medium'):
        configs = {
            'Easy': {'n_ambulances': 3, 'n_patients': 5, 'map_size': 50},
            'Medium': {'n_ambulances': 7, 'n_patients': 15, 'map_size': 100},
            'Hard': {'n_ambulances': 15, 'n_patients': 40, 'map_size': 200}
        }
        config = configs.get(difficulty, configs['Medium'])
        self.n_ambulances = config['n_ambulances']
        self.n_patients = config['n_patients']
        self.map_size = config['map_size']
        self.reset()

    def reset(self) -> Dict:
        self.ambulances = np.random.uniform(0, self.map_size, (self.n_ambulances, 2))
        self.amb_available = np.ones(self.n_ambulances, dtype=bool)
        self.patients = np.random.uniform(0, self.map_size, (self.n_patients, 2))
        self.pat_active = np.ones(self.n_patients, dtype=bool)
        self.pat_wait_times = np.zeros(self.n_patients)
        self.steps = 0
        self.max_steps = self.n_patients * 2
        return self._get_state()

    def _get_state(self) -> Dict:
        return {
            'ambulances': self.ambulances.copy(),
            'available_mask': self.amb_available.copy(),
            'patients': self.patients.copy(),
            'active_mask': self.pat_active.copy(),
            'wait_times': self.pat_wait_times.copy()
        }

    def step(self, action: Optional[Tuple[int, int]]) -> Tuple[Dict, float, bool]:
        reward = 0.0
        self.steps += 1
        self.pat_wait_times[self.pat_active] += 1
        if action:
            amb_idx, pat_idx = action
            if self.amb_available[amb_idx] and self.pat_active[pat_idx]:
                dist = np.linalg.norm(self.ambulances[amb_idx] - self.patients[pat_idx])
                reward += 100 - (dist * 0.2) - (self.pat_wait_times[pat_idx] * 0.5)
                self.ambulances[amb_idx] = self.patients[pat_idx]
                self.amb_available[amb_idx] = False
                self.pat_active[pat_idx] = False
        recovering = (np.random.random(self.n_ambulances) > 0.7) & (~self.amb_available)
        self.amb_available[recovering] = True
        done = not np.any(self.pat_active) or self.steps >= self.max_steps
        return self._get_state(), reward, done
