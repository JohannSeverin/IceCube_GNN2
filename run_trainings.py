import os, sys, tqdm, json, shutil

import os.path as osp


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from scripts.utils import list_experiments, instructions_to_dataset_name
from scripts.train_script import train_model
exp_folder, exp_list = list_experiments()

print(f"Starting process with {len(exp_list)} experiments")

# Loop over the experiments
for i, experiment in enumerate(exp_list):

    # Load construction dictionairy from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    

    # Try to train the model given the construction dict
    # try: 
    train_model(construct_dict)
    shutil.move(osp.join(exp_folder, "done", experiment))
    print(f"Experiment {experiment[:-5]} done \t {experiment}: {i} / {len(exp_list)}")
    # except:
    #     shutil.move(osp.join(exp_folder, "failed", experiment))
    #     print(f"Experiment {experiment[:-5]} failed \t {experiment}: {i} / {len(exp_list)}")


    # Create a script to go through and test the performance
    test_model(model = construct_dict['Experiment'], data = instructions_to_dataset_name(construct_dict))
    




# We can setup a shutdown maybe
#os.system("shutdown -h 5")





    
    


