# ECLIPSE (Fork)

This repository is a fork of the original implementation of the paper "Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer."

## Contributions

This fork includes several enhancements and additional features to the original implementation:

- Support for more models, including 'lexi' and 'llama3-8b-instruct'.
- Advanced functionality for testing prompts using specific CUDA devices for attackers, targets, and judges.
- Enhanced logging capabilities to monitor different aspects of model behavior and the optimization process.
- A framework to load and process specific JSON files containing prompts for evaluation against the models.
- Integration of a custom scoring mechanism to evaluate the effectiveness of generated prompts.

These additions are aimed at providing more flexibility and control over the adversarial suffix optimization process, allowing for more extensive experimentation and analysis.

## Requirements

To install the required packages:

```bash
pip install -r requirements.txt
```

## Running Code

To run the main code, use this command:

```bash
python Eclipse.py --model llama2-7b-chat --dataset 1 --cuda 0 --batchsize 8 --K_round 50 --ref_history 10
```

> 📋 We provide three open-source LLMs ['llama2-7b-chat', 'vicuna-7b', 'falcon-7b-instruct'] here. Dataset 1 is used for comparison with GCG, and Dataset 2 is used for template-based methods. If you want to specify a particular LLM as the attacker, you can add the `--attacker` parameter.

To attack the GPT-3.5-Turbo, run this command:

```bash
python Eclipse-gpt.py --model gpt3.5 --dataset 1 --cuda 0 --batchsize 8 --K_round 50 --ref_history 10
```

## Additional Features

- **Custom Model Support:** Added new models, including 'lexi' and 'llama3-8b-instruct'.
- **Multiple CUDA Devices:** Allows specifying different CUDA devices for attacker, target, and judge models with parameters such as `--attacker_cuda`, `--target_cuda`, and `--judge_cuda`.
- **JSON Prompt Evaluation:** Load and process specific JSON files containing prompts to check their effectiveness against the models.
- **Detailed Logging:** Enhanced logging functionality for better tracking and debugging of model behaviors during optimization rounds.

## Pre-trained Models

You can download pre-trained models here:

- [llama2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf#/)
- [llama3-8b-instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct#)
- [lexi] (https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored)
- [vicuna-7b](https://huggingface.co/lmsys/vicuna-7b-v1.5#/)
- [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct#/)
- [Harmfulness Scorer](https://huggingface.co/hubert233/GPTFuzz#/)

Replace the local model paths in the code file with the downloaded model paths.

## Acknowledgments

This fork builds upon the original work and extends its capabilities to enhance research in adversarial suffix optimization. Special thanks to the original authors for their foundational work. The additional contributions in this fork are aimed at providing greater flexibility, improved performance, and more comprehensive testing frameworks.
