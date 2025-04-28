<!-- Dataset README -->

# Overview
This directory contains the datasets used for profling and evaluation:

1. **Microsoft Azure LLM Inference Trace:** 
    For the evaluation step we use the [Azure LLM inference traces](https://github.com/Azure/AzurePublicDataset/tree/master) which includes: (a) the code (`AzurePublicDataset/data/AzureLLMInferenceTrace_code.csv`), and (b) the conversation (`AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv`) traces.

2. **Dummy Prompt:**
    For the profiling step we use a dummy prompt (`dummy_2048.txt`) that uses the maximum amount of input tokens.
