# 🚑 Emergency Resource Allocation using Reinforcement Learning
"This project focuses on the decision-making intelligence behind emergency systems, not just visualization.”
## 📌 Problem Statement
Efficient allocation of limited emergency resources (like ambulances or medical supplies) is critical in time-sensitive situations.

This project simulates a system where an AI agent must decide:
- Which request to serve
- In what order
- Under time and resource constraints

---

## 🧠 Project Overview

We designed a grid-based environment where:
- Requests appear at different locations
- Each request has a priority (critical / normal)
- The agent must serve them before time runs out

This project models the **decision-making layer** of real-world systems such as:
- Emergency response (ambulances)
- Logistics and delivery
- Hospital resource allocation

---

## ⚙️ Environment Design

- Grid Size: 5 × 5
- Agent starts at: `[0,0]`
- Requests:
  - 🔴 Critical (high priority)
  - 🟢 Normal (low priority)
- Limited:
  - Time
  - Resources

---

## 🎮 Actions

| Action | Description |
|-------|------------|
| 0 | Move Up |
| 1 | Move Down |
| 2 | Move Left |
| 3 | Move Right |
| 4 | Allocate Resource |

---

## 🎯 Reward System

| Event | Reward |
|------|--------|
| Step taken | -0.5 |
| Serve normal request | +10 |
| Serve critical request | +20 |
| All requests completed | +30 |
| Time runs out | -20 |

---

## 🤖 Agents Implemented

### 1. Random Agent
- Takes random actions
- Baseline for comparison

---

### 2. Smart Agent (Nearest)
- Moves toward nearest request
- Improves efficiency

---

### 3. Priority Agent
- Serves critical requests first
- May increase travel cost

---

### 4. Hybrid Agent ⭐
- Balances distance and priority
- Uses:
  - `score = distance - priority_bonus`
- Best performing heuristic agent

---

### 5. Trained Agent (Q-Learning) 🧠
- Learns from experience using rewards
- Builds a Q-table mapping:
  - State → Best action
- Demonstrates reinforcement learning

---

## 📊 Results & Insights

- Random agent performs poorly
- Smart agent improves performance using distance
- Priority agent focuses on urgency but may lose efficiency
- Hybrid agent balances both and performs best
- Trained agent shows learning behavior but depends on state representation

---

## 🔍 Key Insight

> A balance between urgency (priority) and efficiency (distance) is essential in real-world decision-making systems.

---

## 🧠 Reinforcement Learning Aspect

This project demonstrates:
- Environment design (`step`, `reset`)
- Reward-based learning
- Policy comparison
- Q-learning training

---

## 🌐 Real-World Mapping

| System Component | Real World |
|----------------|-----------|
| Requests | Emergency calls / app requests |
| Location | GPS coordinates |
| Resources | Ambulances / vehicles |
| Agent | Decision system (AI) |

---

## 🚀 How to Run

```bash
pip install gradio numpy
python app.py
