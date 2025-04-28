import socket
import threading
import queue
import argparse
import ast
import time
import csv
import subprocess
import signal
import os
from pathlib import Path
from mpi4py import MPI
import numpy as np
import torch
import signal
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)
from MS_utils import send_msg, recv_msg, recvall
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file
sys.path.append(f'{BASE_DIR}/../../TensorRT-LLM/tensorrt_llm')

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

OUTPUTS_DIR = f'{BASE_DIR}/outputs'

# Check whether 'OUTPUTS_DIR' directory exists, otherwise create it
if not os.path.exists(f'{OUTPUTS_DIR}'): 
    os.makedirs(f'{OUTPUTS_DIR}')
    print(f"Directory created: {OUTPUTS_DIR}")

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--client_id',
                        type=str,
                        help="The client id")
    parser.add_argument('--output_mode',
                        type=str,
                        help="Output to txt or send response back",
                        default='response')

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
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )

    return parser.parse_args(args=args)

def timeout_handler(signum, frame):
    raise TimeoutError("Function has timed out")

def load_model(args, runtime_rank):
    '''
    Load the LLM using tensorRT-LLM
    '''

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

    runner = runner_cls.from_dir(**runner_kwargs)

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                input_text='WARMUP',
                                prompt_template=prompt_template,
                                input_file=args.input_file,
                                add_special_tokens=args.add_special_tokens,
                                max_input_length=args.max_input_length,
                                pad_id=pad_id,
                                num_prepend_vtokens=args.num_prepend_vtokens,
                                model_name=model_name,
                                model_version=model_version)

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

    return runner, tokenizer, prompt_template, pad_id, end_id 

def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None,
                as_serving=False):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if (input_file is None) or as_serving:
        if isinstance(input_text, list):
            for curr_text in input_text:
                if prompt_template is not None:
                    curr_text = prompt_template.format(input_text=curr_text)
                input_ids = tokenizer.encode(curr_text,
                                            add_special_tokens=add_special_tokens,
                                            truncation=True,
                                            max_length=max_input_length)
                batch_input_ids.append(input_ids)
        else:
            curr_text = input_text
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
                print(f'Input Tokenized length: {len(input_ids)}')
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


def output_generation(args, batch_input_ids, runner, bad_words_list, stop_words_list, pad_id, end_id, timeout):
    signal.alarm(timeout)
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
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

            torch.cuda.empty_cache()
    finally:
        # Disable the alarm
        signal.alarm(0)
    return outputs

def inference_task(args, inference_queue, result_queue, runner, tokenizer, prompt_template, pad_id, end_id, comm, timeout = 150):
    runtime_rank = tensorrt_llm.mpi_rank()
    should_continue = False
    try:
        if runtime_rank == 0:
            host = socket.gethostname()
            port = 5000
            client_id = args.client_id
            client_socket.connect((host, port))
            #client_socket.sendall(client_id.encode())
            send_msg(client_socket, client_id.encode())
            print(f'{client_id}: Connected to server on {host}:{port}')

        #threading.Thread(target=sending_outputs, args=(args, result_queue, client_socket), daemon=True).start()

        while True:
        
            if runtime_rank == 0:
                #data = client_socket.recv(8192).decode()
                data = recv_msg(client_socket).decode()
                if not data:
                    print(f'{client_id}: Server closed the connection.')
                    break

                print(f'{client_id}: Received from server:', data)

                if data == 'exit':
                    current_pid = os.getpid()
                    os.kill(current_pid, signal.SIGTERM)
                #inference_queue.put(data)
                #data = inference_queue.get() 
                try:
                    req_id, start_time, inputs = data.split(maxsplit=2)
                except ValueError as e:
                    sending_outputs(args, 'EMPTY INPUT')
                #inputs = ast.literal_eval(inputs)
                #print(f'input!:{inputs}')
                if inputs == None:
                    sending_outputs(args, 'EMPTY INPUT')
                #inputs = inputs[0:500]
                #print(f'input!:{inputs}')
                if start_time == '0':
                    start_time = time.time()
            else:
                inputs = None

            inputs = comm.bcast(inputs, root=0)

            #print(f"Get gogo {inputs}")

            bad_words_list = None
            stop_words_list = None

            model_name, model_version = read_model_name(args.engine_dir)
            batch_input_ids = parse_input(tokenizer=tokenizer,
                                        input_text=inputs,
                                        prompt_template=prompt_template,
                                        input_file=args.input_file,
                                        add_special_tokens=args.add_special_tokens,
                                        max_input_length=args.max_input_length,
                                        pad_id=pad_id,
                                        num_prepend_vtokens=args.num_prepend_vtokens,
                                        model_name=model_name,
                                        model_version=model_version,
                                        as_serving=True)
            input_lengths = [x.size(0) for x in batch_input_ids]
            torch.cuda.synchronize()
            
            if runtime_rank == 0:
                print(f'input!:{batch_input_ids}')
                #tensorrt_llm.profiler.start("tmp")
                

            try:
                this_time = time.time()
                outputs = output_generation(args, batch_input_ids, runner, bad_words_list, stop_words_list, pad_id, end_id, timeout)
                end_time = time.time()
            except TimeoutError as e:
                if runtime_rank == 0:
                    result_queue.put('TIMEOUT ERROR')
                continue

            #print("[Client Start] Performing inference on data:", inputs)
            #if runtime_rank == 0:
                #tensorrt_llm.profiler.start("tmp")
                

            if runtime_rank==0:
                
                print(
                        f"batch_size: {len(batch_input_ids)}, latency: {tensorrt_llm.profiler.elapsed_time_in_sec('tmp')} sec, but I think it is {end_time - this_time} sec"
                    )

                output_ids = outputs['output_ids']
                sequence_lengths = outputs['sequence_lengths']
                context_logits = None
                generation_logits = None
                if runner.gather_context_logits:
                    context_logits = outputs['context_logits']
                if runner.gather_generation_logits:
                    generation_logits = outputs['generation_logits']
                
                batch_size, num_beams, _ = output_ids.size()

                #print('inference finished')

                for batch_idx in range(batch_size):
                    inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
                    )
                    input_text = tokenizer.decode(inputs)
                    
                    #print(f'Input [Text {batch_idx}]: \"{input_text}\"')
                    for beam in range(num_beams):
                        output_begin = input_lengths[batch_idx]
                        output_end = sequence_lengths[batch_idx][beam]
                        outputs = output_ids[batch_idx][beam][
                            output_begin:output_end].tolist()
                        output_text = tokenizer.decode(outputs)
                        torch.cuda.synchronize()
                        #print(
                        #    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"') 
                        #end_time = time.time()
                        result = [f'{req_id}',f'{end_time - float(start_time)}',f'{end_time - this_time}',output_text.replace('\n', 'CHANGELINE').replace(',','COMMA')]
                        #result = req_id + ',' + start_time + ','+ f'{end_time}' + ',' + output_text
                        #print(result)
                        #result_queue.put(result)
                        sending_outputs(args, result)

    except ConnectionRefusedError:
        if runtime_rank==0:
            print(f'{client_id}: Failed to connect to the server. Please make sure the server is running.')
    finally:
        if runtime_rank==0:
            client_socket.close()
            inference_queue.put("exit")


def sending_outputs(args, result):
    global client_socket

    #while True:
    #if not result_queue.empty():
    #    result = result_queue.get()
        #output mode
    if args.output_mode == 'response':    
        send_msg(client_socket, ','.join(result).encode())
        #client_socket.send(','.join(result).encode())
    elif args.output_mode == 'txt': 
        send_msg(client_socket, 'finished'.encode())
        #client_socket.send('finished'.encode())
        with open(f'{OUTPUTS_DIR}/{args.client_id}_output_records.csv', 'a') as file:
            writer = csv.writer(file, escapechar='\\', quoting=csv.QUOTE_NONE)
            formatted_string = [args.client_id] + result
            writer.writerow(formatted_string)  
                
                                   


def io_thread(args, client_id, host, port, inference_queue, result_queue):
    global client_socket
    
    try:
        client_socket.connect((host, port))
        send_msg(client_socket, client_id.encode())
        #client_socket.sendall(client_id.encode())
        print(f'{client_id}: Connected to server on {host}:{port}')

        #threading.Thread(target=sending_outputs, args=(args, result_queue, client_socket), daemon=True).start()

        while True:
            #data = client_socket.recv(819200).decode()
            data = recv_msg(client_socket).decode()
            if not data:
                print(f'{client_id}: Server closed the connection.')
                break

            print(f'{client_id}: Received from server:', data)

            if data == 'exit':
                #current_pid = os.getpid()
                #os.kill(current_pid, signal.SIGTERM)
                sys.exit()
            inference_queue.put(data)

    except ConnectionRefusedError:
        print(f'{client_id}: Failed to connect to the server. Please make sure the server is running.')
    finally:
        client_socket.close()
        inference_queue.put("exit")

def client_program(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    comm = MPI.COMM_WORLD

    # Load LLM
    #runner, tokenizer, prompt_template, pad_id, end_id = load_model(args, runtime_rank)

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

    runner = runner_cls.from_dir(**runner_kwargs)

    batch_input_ids = parse_input(tokenizer=tokenizer,
                                input_text='WARMUP',
                                prompt_template=prompt_template,
                                input_file=args.input_file,
                                add_special_tokens=args.add_special_tokens,
                                max_input_length=args.max_input_length,
                                pad_id=pad_id,
                                num_prepend_vtokens=args.num_prepend_vtokens,
                                model_name=model_name,
                                model_version=model_version)
    
    if runtime_rank == 0:
        time1 = time.time()

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

    if runtime_rank == 0:
        time2 = time.time()
        #print(time2-time1)
        #print(outputs)
    

    inference_queue = queue.Queue()
    result_queue = queue.Queue()
    #threading.Thread(target=inference_task, args=(inference_queue, result_queue, runner, tokenizer, prompt_template, pad_id, end_id, comm), daemon=True).start()

    if runtime_rank == 0:

        if args.output_mode == 'txt':  
            with open(f'{OUTPUTS_DIR}/{args.client_id}_output_records.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['client_id','req_id','req_time','complete_time','output'])

        #threading.Thread(target=io_thread, args=(args, args.client_id, host, port, inference_queue, result_queue), daemon=True).start()
        

    inference_task(args, inference_queue, result_queue, runner, tokenizer, prompt_template, pad_id, end_id, comm)
        #io_thread(args.client_id, host, port, inference_queue, result_queue)

if __name__ == '__main__':
    args = parse_arguments()
    client_program(args)

