import os
import torch
import gradio as gr
from typing import List, Tuple
from reverse_abliterator import ReverseAbliterator

def get_available_models(base_dir: str) -> List[str]:
    models = []
    # Go up two directories from the current working directory to reach the model_git directory
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    
    # List all directories in the root_dir
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            # Check if the directory contains any .safetensors files
            if any(file.endswith('.safetensors') for file in os.listdir(dir_path)):
                models.append(dir_name)
    
    return models

# Update the base_dir assignment
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
available_models = get_available_models(base_dir)

# Print available models for debugging
print("Available models:", available_models)

def perform_reverse_ablation(
    model_path: str,
    target_instructions: str,
    baseline_instructions: str,
    enhancement_strength: float,
    num_test_samples: int,
    max_tokens_generated: int
) -> str:
    full_model_path = os.path.join(base_dir, model_path)
    # Prepare the dataset
    target_instr = target_instructions.split('\n')
    baseline_instr = baseline_instructions.split('\n')
    dataset = ([target_instr, baseline_instr])

    # Initialize ReverseAbliterator
    reverse_abliterator = ReverseAbliterator(
        model=full_model_path,
        dataset=dataset,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Cache activations
    reverse_abliterator.cache_activations(N=len(target_instr), batch_size=1)

    # Measure initial enhancement
    initial_enhancement = reverse_abliterator.measure_enhancement()
    
    # Enhance the model
    reverse_abliterator.enhance_model(strength=enhancement_strength)

    # Measure enhancement after modification
    post_enhancement = reverse_abliterator.measure_enhancement()

    # Test the enhanced model
    test_results = []
    for prompts in reverse_abliterator.target_inst_test[:num_test_samples]:
        toks = reverse_abliterator.tokenize_instructions_fn([prompts])
        _, all_toks = reverse_abliterator.generate_logits(toks, max_tokens_generated=max_tokens_generated)
        response = reverse_abliterator.model.tokenizer.decode(all_toks[0], skip_special_tokens=True)
        test_results.append(f"Prompt: {prompts}\nResponse: {response}\n")

    # Save the modified model
    new_model_name = os.path.basename(model_path) + "_anti-ablated"
    new_model_path = os.path.join(os.path.dirname(os.path.dirname(model_path)), new_model_name)
    os.makedirs(new_model_path, exist_ok=True)
    reverse_abliterator.save_activations(os.path.join(new_model_path, "enhanced_model_state.pt"))

    # Prepare the output
    output = f"Initial enhancement score: {initial_enhancement['enhancement'].item():.4f}\n"
    output += f"Post-enhancement score: {post_enhancement['enhancement'].item():.4f}\n\n"
    output += "Test Results:\n" + "\n".join(test_results)
    output += f"\nEnhanced model saved to: {new_model_path}"

    return output

# Set up the Gradio interface
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
# base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
available_models = get_available_models(base_dir)

iface = gr.Interface(
    fn=perform_reverse_ablation,
    inputs=[
        gr.Dropdown(choices=available_models, label="Select Model"),
        gr.Textbox(lines=5, label="Target Instructions (one per line)"),
        gr.Textbox(lines=5, label="Baseline Instructions (one per line)"),
        gr.Slider(minimum=0.01, maximum=1.0, step=0.01, value=0.1, label="Enhancement Strength"),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of Test Samples"),
        gr.Slider(minimum=10, maximum=100, step=10, value=30, label="Max Tokens Generated"),
    ],
    outputs=gr.Textbox(label="Results", lines=10),
    title="Reverse Abliterator",
    description="Select a model and perform reverse ablation to enhance specific capabilities."
)

if __name__ == "__main__":
    iface.launch()
