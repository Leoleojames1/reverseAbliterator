# Reverse Abliteration: Advanced Technique for AI Model Modification🧠🔧

[![PyPI version](https://badge.fury.io/py/reverse-abliterator.svg)](https://badge.fury.io/py/reverse-abliterator)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub issues](https://img.shields.io/github/issues/leoleojames1/reverse-abliterator)](https://github.com/leoleojames1/reverse-abliterator/issues)

## Introduction 🌟

This project explores abliteration and reverse abliteration, two innovative techniques for modifying pre-trained language models. These methods offer targeted approaches to altering model behavior, complementing traditional fine-tuning techniques. This tool is a reverse construction of FailSpy's abliterator repo which you can check out here:
[FailSpy/abliterator](https://github.com/FailSpy/abliterator)

## Core Concepts 💡

1. Abliteration suppresses the desired style directions.
2. Reverse Abliteration amplifies the desired style directions.

### Abliteration 🚫

Abliteration is designed to selectively suppress specific behaviors or capabilities of a language model while preserving its overall functionality.

Key features:
- Aims to reduce or eliminate undesired outputs or behaviors 🛑
- Modifies model weights by subtracting projections along refusal directions in the model's activation space 📉
- Preserves general model capabilities while targeting specific behaviors for suppression 🎯

### Reverse Abliteration 🔄

Reverse abliteration, conversely, aims to enhance or amplify certain behaviors or capabilities of a language model.

Key features:
- Seeks to strengthen or improve specific desired outputs or behaviors 📈
- Modifies model weights by adding projections along style or enhancement directions in the model's activation space 🔼
- Enhances targeted capabilities while aiming to minimally impact other functionalities 🎭

## Mechanism ⚙️

Both techniques leverage a similar process:

1. Activation Caching: Run instructions through the model and store activations at specified layers. 💾
2. Direction Calculation: Compute the difference between mean activations of target and baseline instructions. 🧮
3. Weight Modification: Modify model weights by projecting onto calculated directions. 🔧
4. Iterative Refinement: Apply modifications iteratively, testing performance after each step. 🔁

The key difference lies in the final step:
- Abliteration subtracts the projection to suppress behaviors ➖
- Reverse abliteration adds the projection to enhance behaviors ➕

## Comparison with Traditional Fine-tuning 🔍

1. Granularity:
   - Fine-tuning: Broad adjustment of model behavior 🌊
   - Abliteration/Reverse Abliteration: Targeted modification of specific behaviors 🎯

2. Data Requirements:
   - Fine-tuning: Substantial dataset for the target task 📚
   - Abliteration/Reverse Abliteration: Can work with smaller, curated datasets 📊

3. Computational Efficiency:
   - Fine-tuning: Often computationally intensive 🖥️💨
   - Abliteration/Reverse Abliteration: Can be more efficient, focusing on specific components ⚡

4. Preservation of Capabilities:
   - Fine-tuning: Risk of catastrophic forgetting 🧠💨
   - Abliteration/Reverse Abliteration: Designed to preserve general capabilities 🛡️

5. Interpretability:
   - Fine-tuning: Changes can be difficult to interpret 🕵️‍♀️
   - Abliteration/Reverse Abliteration: More targeted and potentially more interpretable modifications 🔬

## Applications 🚀

### Abliteration:
- Reducing biases and harmful outputs 🚯
- Creating "safer" versions of models 🛡️
- Consistently avoiding certain types of outputs 🚫

### Reverse Abliteration:
- Enhancing specific skills (e.g., mathematical reasoning, creative writing) 🧮✍️
- Specializing models for particular domains 👩‍⚕️👨‍💼👩‍🔬
- Injecting specific writing styles or personas 🎭
- Improving emotional intelligence in responses 🤗
- Adapting models to new tasks or contexts quickly 🦾

## Advantages and Limitations ⚖️

Advantages:
- Targeted modifications with potential for better interpretability 🎯🔍
- Computational efficiency compared to full fine-tuning ⚡
- Potential for better preservation of general capabilities 🛡️

Limitations:
- Requires careful identification of relevant directions 🧭
- Potential for unexpected side effects 🎲
- Less suitable for broad changes to model behavior 🌊

## Considerations 🤔

1. Stability and Generalization: Enhancements may lead to overfitting or unexpected effects. 🌋
2. Ethical Implications: Rapid behavior modification raises important ethical questions. 🧑‍⚖️
3. Validation and Testing: Robust frameworks are crucial for ensuring reliability and safety. 🧪
4. Theoretical Foundations: Further research is needed on how these modifications affect internal representations. 📚🔬
5. Combination with Other Techniques: Potential for powerful hybrid approaches with other training methods. 🧩

## Conclusion 🌟

Abliteration and reverse abliteration represent sophisticated approaches to selectively modifying language model behavior. By offering more targeted and potentially more efficient ways to adjust model behavior, these methods open up new possibilities for customizing and improving AI systems. As research progresses, these techniques may lead to more adaptable, specialized, and capable AI models, while also presenting important challenges in terms of stability, ethics, and validation that need careful consideration.

These innovative approaches complement traditional fine-tuning, offering a spectrum of model modification techniques that can be tailored to specific needs and constraints. As the field of AI continues to evolve, abliteration and reverse abliteration stand as promising tools for creating more nuanced and controllable AI systems. 🚀🤖

## Installation

You can install the Reverse Abliterator package using pip:

```
pip install reverseAbliterator
```

## Usage

Here's a basic example of how to use the Reverse Abliterator:

```python
from reverseAbliterator import ReverseAbliterator

model_path = "path/to/your/model"
target_instructions = ["Write a poem about nature", "Explain quantum physics"]
baseline_instructions = ["Hello", "What's the weather like?"]

ra = ReverseAbliterator(
    model=model_path,
    dataset=([target_instructions, baseline_instructions]),
    device="cuda" if torch.cuda.is_available() else "cpu"
)

ra.cache_activations(N=len(target_instructions), batch_size=1)
initial_enhancement = ra.measure_enhancement()
print("Initial enhancement score:", initial_enhancement)

ra.enhance_model(strength=0.1)
post_enhancement = ra.measure_enhancement()
print("Post-enhancement score:", post_enhancement)

ra.test_enhancement(N=2, max_tokens_generated=30)

# Save the modified model
ra.save_modified_model("path/to/save/modified_model.pth")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

