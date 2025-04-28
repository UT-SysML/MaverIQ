<!-- Models README -->

# Overview
This directory contains the models's weight and configuration files the instrustions to donwload them for the HuggingFace library are as follows:

0. **GIT Packages Installation:**
```bash
apt-get update && apt-get install -y git git-lfs
```

1. **Falcon Models:**
```bash
# Setup git-lfs
git lfs install

# falcon-7b-instruct
git clone https://huggingface.co/tiiuae/falcon-7b-instruct ./falcon/falcon-7b

# falcon-40b-instruct
git clone https://huggingface.co/tiiuae/falcon-40b-instruct ./falcon/falcon-40b
```

2. **GPT-J-6B Models:**
```bash
# Setup git-lfs
git lfs install

# gptj-6b
git clone https://huggingface.co/EleutherAI/gpt-j-6b ./gptj/gptj-6b
```

3. **Llama-2 Models:**
The Llama-2 models are governed by the Meta license. In order to download the model weights and tokenizer, follow the instructions in the HuggingFace website:

- Llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b
- Llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b
- Llama-2-70b: https://huggingface.co/meta-llama/Llama-2-70b-hf


# Directory's Structure
The directory should have the following structure:

```bash
|-- models              : Directory that contains the models` weight and configuration files
|   |-- falcon          : Falcon Models
|   |   |-- falcon-7b
|   |   |-- falcon-40b
|   |-- gptj            : GPT-J Model
|   |   |-- gptj-6b
|   |-- llama-2         : Llama-2 Models
|   |   |-- llama-2-7b
|   |   |-- llama-2-13b
|   |   |-- llama-2-70b
```