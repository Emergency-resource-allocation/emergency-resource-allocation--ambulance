import gradio as gr
from main import run_simulation

def run():
    result = run_simulation()
    return "Simulation completed successfully!"

demo = gr.Interface(
    fn=run,
    inputs=[],
    outputs="text",
    title="Emergency Resource Allocation AI",
    description="Run simulation for ambulance allocation using RL"
)

demo.launch()