# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import csv
import subprocess
import signal
import os
from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

import time
from pynvml import *
import pandas as pd

""" Helper functions """
################################################################################################################################
def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

def calc_GPU_info(dev):
    if torch.cuda.is_available():
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(dev)
        info = nvmlDeviceGetMemoryInfo(h)
        return bytes_to_giga_bytes(info.total), bytes_to_giga_bytes(info.free), bytes_to_giga_bytes(info.used) #return GB
    else:
        return 0, 0, 0
################################################################################################################################

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_utilization',
                        default=False,
                        action='store_true',
                        help="Record GPU utilization and memory usage.")
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument(
        '--file_size',
        type=int,
        help='Number of inputs (batch size) of given file',
        default=1
    )
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default='gpt2')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens.")
    parser.add_argument(
        '--save_mem_profiling_data',
        default=False,
        action='store_true',
        help="Save memory profiling data in GPU_mem.csv")

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        prompt_count = args.file_size
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader, None)
                for line in csv_reader:
                    prompt_count -= 1
                    input_ids = tokenizer.encode(
                                line[0],
                                add_special_tokens=add_special_tokens,
                                truncation=True,
                                max_length=max_input_length)
                    batch_input_ids.append(input_ids[-max_input_length:])
                    if prompt_count == 0:
                        break
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                prompt_count -= 1
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
                if prompt_count == 0:
                    break                
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.read()
                input_ids = tokenizer.encode(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)
                batch_input_ids.append(input_ids)
                batch_input_ids = batch_input_ids[:prompt_count]
        else:
            print('Input file format not supported.')
            raise SystemExit
    if model_name == 'GemmaForCausalLM':
        batch_input_ids[0] = [tokenizer.bos_token_id] + batch_input_ids[0]

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 output_logits_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(
                    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    # Save context logits
    if context_logits is not None and output_logits_npy is not None:
        context_logits = torch.cat(context_logits, axis=0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])

        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)

    # Save generation logits
    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)


def main(args):

    if args.save_mem_profiling_data:
        if tensorrt_llm.mpi_rank() == 0:
            GPU_mem = {'init':{'total':[], 'free':[], 'used':[]},
                       'pre_run':{'total':[], 'free':[], 'used':[]},
                       'after_run':{'total':[], 'free':[], 'used':[]}}

    if tensorrt_llm.mpi_rank() == 0:
        print("GPU INFO INIT")
        for GPU_num in range(torch.cuda.device_count()):
            total, free, used = calc_GPU_info(GPU_num)
            print(f"GPU_{GPU_num} Info: Total - {total:.2f} GB, Free - {free:.2f} GB, Used - {used:.2f} GB")
            if args.save_mem_profiling_data:
                GPU_mem['init']['total'].append(total)
                GPU_mem['init']['free'].append(free)
                GPU_mem['init']['used'].append(used)


    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
    # stop_words_list = [[" London"], ["eventually became"]]
    # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
    # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
    stop_words_list = None

    # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
    # bad_words_list = [[" chef"], [" eventually, chef before"]]
    # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
    # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
    bad_words_list = None

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]
    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  prompt_template=prompt_template,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name,
                                  model_version=model_version)
    input_lengths = [x.size(0) for x in batch_input_ids]

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(engine_dir=args.engine_dir,
                         lora_dir=args.lora_dir,
                         rank=runtime_rank,
                         debug_mode=args.debug_mode,
                         lora_ckpt_source=args.lora_ckpt_source)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.use_py_session, "Medusa is only supported by py_session"
        assert args.temperature == 0, "Medusa should use temperature == 0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=len(batch_input_ids),
            max_input_len=max(input_lengths),
            max_output_len=args.max_output_len,
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
        )
    runner = runner_cls.from_dir(**runner_kwargs)

    if tensorrt_llm.mpi_rank() == 0:
        print("GPU INFO PRE RUN")
        for GPU_num in range(torch.cuda.device_count()):
            total, free, used = calc_GPU_info(GPU_num)
            print(f"GPU_{GPU_num} Info: Total - {total:.2f} GB, Free - {free:.2f} GB, Used - {used:.2f} GB")
            if args.save_mem_profiling_data:
                GPU_mem['pre_run']['total'].append(total)
                GPU_mem['pre_run']['free'].append(free)
                GPU_mem['pre_run']['used'].append(used)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids,
            max_new_tokens=args.max_output_len,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            lora_uids=args.lora_task_uids,
            prompt_table_path=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            return_dict=True,
            medusa_choices=args.medusa_choices)
        torch.cuda.synchronize()

    if tensorrt_llm.mpi_rank() == 0:
        print("GPU INFO AFTER RUN")
        for GPU_num in range(torch.cuda.device_count()):
            total, free, used = calc_GPU_info(GPU_num)
            print(f"GPU_{GPU_num} Info: Total - {total:.2f} GB, Free - {free:.2f} GB, Used - {used:.2f} GB")
            if args.save_mem_profiling_data:
                GPU_mem['after_run']['total'].append(total)
                GPU_mem['after_run']['free'].append(free)
                GPU_mem['after_run']['used'].append(used)
    
    if args.save_mem_profiling_data:
        if tensorrt_llm.mpi_rank() == 0:
            # Convert the nested dictionary to a DataFrame
            df = pd.DataFrame.from_dict({(i,j): GPU_mem[i][j] 
                                        for i in GPU_mem.keys() 
                                        for j in GPU_mem[i].keys()},
                                        orient='index')

            # Save the DataFrame to a CSV file
            df.to_csv('GPU_mem.csv')


    if args.streaming:
        for curr_outputs in throttle_generator(outputs,
                                               args.streaming_interval):
            if runtime_rank == 0:
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                print_output(tokenizer,
                             output_ids,
                             input_lengths,
                             sequence_lengths,
                             output_csv=args.output_csv,
                             output_npy=args.output_npy)
    else:
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            context_logits = None
            generation_logits = None
            if runner.gather_context_logits:
                context_logits = outputs['context_logits']
            if runner.gather_generation_logits:
                generation_logits = outputs['generation_logits']
            print_output(tokenizer,
                         output_ids,
                         input_lengths,
                         sequence_lengths,
                         output_csv=args.output_csv,
                         output_npy=args.output_npy,
                         context_logits=context_logits,
                         generation_logits=generation_logits,
                         output_logits_npy=args.output_logits_npy)

    if args.run_profiling:
        ite = 3
        # warmup
        for _ in range(2):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    lora_uids=args.lora_task_uids,
                    prompt_table_path=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True)
                torch.cuda.synchronize()

        pid = -1
        print('warm up ends')

        if args.record_utilization:
            command_record = [
                "nvidia-smi",
                "--query-gpu=timestamp,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                "--format=csv",
                "-l", "1",
                "-f", f"./GPU-{str(runtime_rank)}-record-{args.tokenizer_dir.replace('/', '')}-.csv"
                ]
                
            process = subprocess.Popen(command_record) 
            pid = process.pid
            #print(f'the PID is: {pid}')

        tensorrt_llm.profiler.start("tmp")
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    lora_uids=args.lora_task_uids,
                    prompt_table_path=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True)

                #print(tensorrt_llm.profiler.device_memory_info())
                torch.cuda.synchronize()
        tensorrt_llm.profiler.stop("tmp")

        if args.record_utilization:
            os.kill(pid, signal.SIGINT)

        print(
            f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
        )

        if tensorrt_llm.mpi_rank() == 0:
            print(f"Inference Time Data: {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
