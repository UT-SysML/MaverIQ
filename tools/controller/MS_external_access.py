import time
import sched
import csv
import threading
import os
import sys
from MS_controller import inference_main, external_inputs, check_model_load, the_exit, set_inference_stop, set_req_id

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of this file

OUTPUTS_DIR = f'{BASE_DIR}/outputs' # Path to outputs directory
CONTROLLER_DIR = f'{BASE_DIR}' # Path to MaverIQ Controller

# Check whether 'OUTPUTS_DIR' directory exists, otherwise create it
if not os.path.exists(f'{OUTPUTS_DIR}'): 
    os.makedirs(f'{OUTPUTS_DIR}')
    print(f"Directory created: {OUTPUTS_DIR}")

count = 0 # Counter to track requests

def execute_command(command):
    '''
    Function to execute command

    Input:
        - command (str): Input Command

    Output:
        - No Output
    '''
    global count

    # Midpoint tool
    if command == 'stop':
        set_inference_stop(True)
    elif command == 'start':
        set_inference_stop(False)
    elif isinstance(command, int):
        set_req_id(command - 1)
    else:
        parts = command.split()

        if ((parts[1] == 'inference') or (parts[1] == 'prior_inference')):
            count += 1

            if len(command) <= len("1 inference "):
                command = command[0] + ' inference Hello!'

            parts = command.split(maxsplit = 2)

            if len(parts) < 2:
                return
            usr_id = parts[0]
            cmd = parts[1]

            if ((cmd == 'inference') or (cmd == 'prior_inference')):

                req_time = str(time.time())
                try:
                    command = " ".join([parts[0],parts[1],req_time,parts[2]])
                except IndexError:
                    command = " ".join([parts[0],parts[1],req_time,'Hello'])
        
        # # Profiling command
        # elif ((parts[0] == 'register') or (parts[0] == 'deploy')):
        else:
            command = command

        # # # Resolve stall
        # if count in [750]:
        #    return
        
        # solve bug
        
        # count2 = 0
        # if parts[1] != 'deploy':
        #     try:
        #         model_id = int(parts[0])
        #         while check_model_load(model_id) is False:
        #             time.sleep(0.5)
        #             count2 += 1
        #             #if count2 > 2000:
        #                 #print("Terminating process due to timeout.")
        #                 #sys.exit()
            
        #     except ValueError:
        #         model_id = 0
        
        print(f'[TRACE INFO] Executing {count} at {time.time()}: {command}')

        external_inputs(command)


def rename_output(args, client_id):
    '''
    Function to rename the generate output files

    Input:
        - args (dictionary): Arguments of Input Command
        - client_id (int): CLient's (Model's) ID

    Output:
        - No Output
    '''
    the_time = time.time()
    file_ext = '-FP16' if args.use_float16_only else ''
    old_file_name = f'{OUTPUTS_DIR}/{client_id}_output_records.csv'
    new_file_name = f'{OUTPUTS_DIR}/MaverIQ{file_ext}_{args.model_list}_{args.load_strategy}_{args.packing_threshold}_{args.scale_factor}_{args.cost_func}_{args.trace}_{args.slo}_{client_id}_output_records.csv'
    if os.path.exists(old_file_name):
        os.rename(old_file_name, new_file_name)
        print("File renamed successfully.")
    else:
        print("Error: The file does not exist.")   


def trace_replay(args, model_list, trace_list):
    '''
    Function to play the trace using a scheduler

    Input:
        - args (dictionary): Arguments of Input Command
        - model_list (list): List of model to be used
        - trace_list (list of dict): The trace

    Output:
        - No Output
    '''
    save_model = True
    external_input = True
    packing_threshold = args.packing_threshold
    print(f"Set packing_threshold to {packing_threshold}")
    threading.Thread(target = inference_main, args=(save_model, external_input,packing_threshold,), daemon=True).start()

    time.sleep(1)

    # Execute Trace line by line
    scheduler = sched.scheduler(time.time, time.sleep)
    for element in trace_list:
        print(element)
        scheduler.enter(element["Timestamp"], 1, execute_command, argument=(element["Command"],))

    scheduler.run()

    #Give some time for results (5-mins)
    time.sleep(300)
    print('Stop Waiting...')

    for i in model_list:
        external_inputs(f"{i['id']} remove")    

    time.sleep(20)

    for model in model_list:
        rename_output(args, model['id'])

    the_exit()


def load_and_inference(client_id, model_name, user_intent='min_lat', output_length = 20):
    '''
    Function to load the model and serve an example request

    Input:
        - client_id (int): Mpdel's ID
        - model_name (str): Model's name
        - output_length: Selected output lenght

    Output:
        - No Output
    '''
    external_inputs(f'{str(client_id)} deploy {model_name} {str(output_length)} {str(user_intent)}')

    while check_model_load(client_id) is False:
        time.sleep(1)
    
    external_inputs(f'{str(client_id)} inference Do you hear the people sing?')

    time.sleep(10)


def main():

    save_model = True
    external_input = True
    threading.Thread(target = inference_main, args=(save_model, external_input,), daemon=True).start()

    time.sleep(1)

    client_id = 1

    model_list_file = f'{CONTROLLER_DIR}/model_load_sequence.csv'
    with open(model_list_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            #print(row)
            load_and_inference(client_id, ''.join(row))
            client_id += 1

    time.sleep(30)

    for i in range(1,client_id):
        external_inputs(f'{i} remove')

if __name__ == '__main__':
    main()