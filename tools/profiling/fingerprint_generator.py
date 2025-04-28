import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import gc
import copy
import shutil

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True,
                        help='The name of the original model (eg. "llama-2-7b")')
    parser.add_argument('--model_parent_dir', 
                        type=str, 
                        required=True,
                        help='The parent directory of the original model')
    parser.add_argument('--model_dir', 
                        type=str, 
                        required=True,
                        help='The directory of the original model')
    parser.add_argument('--num_layers',
                        type=int,
                        required=True,
                        help='The number of hidden layers of the fingerprint')
    return parser.parse_args(args=args)

def flush():
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.reset_peak_memory_stats()

def deleteDecodingLayers(model_name, model, num_layers_to_keep):  # must pass in the full bert model

    if 'llama-2' in model_name:
        oldModuleList = model.model.layers
    elif ('gptj' in model_name) or ('falcon' in model_name):
        oldModuleList = model.transformer.h
    newModuleList = torch.nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    if 'llama-2' in model_name:
        copyOfModel.model.layers = newModuleList
    elif ('gptj' in model_name) or ('falcon' in model_name):
        copyOfModel.transformer.h = newModuleList

    return copyOfModel

def main(args):

    assert (('gptj' not in args.model_name) or ('falcon' not in args.model_name) or ('llama-2' not in args.model_name)), "This model is not supported yet. Supported list:[GPT-J, FALCON, LLAMA-2]"

    model = AutoModelForCausalLM.from_pretrained(f'{args.model_dir}', torch_dtype=torch.float16).to('cpu')
    # tokenizer = AutoTokenizer.from_pretrained(f'{args.model_dir}')

    print(f'\nModel Overview:\n{model}\n')

    fingerprint = deleteDecodingLayers(args.model_name, model, args.num_layers)
    print(f'\nNew Model Overview:\n{fingerprint}\n')

    if ('gptj' in args.model_name):
        fingerprint.config.n_layer = args.num_layers
    elif ('llama-2' in args.model_name) or ('falcon' in args.model_name):
        fingerprint.config.num_hidden_layers = args.num_layers
    fingerprint.save_pretrained(f"{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/", from_pt=True)

    #Copy the Python files for falcon and llama-2
    if ('llama-2' in args.model_name):
        shutil.copy2(f'{args.model_dir}/special_tokens_map.json', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/special_tokens_map.json')
        shutil.copy2(f'{args.model_dir}/tokenizer_config.json', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/tokenizer_config.json')
        shutil.copy2(f'{args.model_dir}/tokenizer.json', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/tokenizer.json')
        shutil.copy2(f'{args.model_dir}/tokenizer.model', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/tokenizer.model')
    elif ('falcon' in args.model_name):
        shutil.copy2(f'{args.model_dir}/configuration_falcon.py', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/configuration_falcon.py')
        shutil.copy2(f'{args.model_dir}/handler.py', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/handler.py')
        shutil.copy2(f'{args.model_dir}/modeling_falcon.py', f'{args.model_parent_dir}/{args.model_name}-{args.num_layers}layer/modeling_falcon.py')


if __name__ == "__main__":
    args = parse_arguments()
    main(args)