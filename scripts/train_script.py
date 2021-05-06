import tqdm, os, sys, time, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

import numpy as np

import os.path as osp

from tensorflow.keras.optimizers import Adam
from spektral.data import DisjointLoader
from importlib import __import__

file_path = osp.dirname(osp.realpath(__file__))

def train_model(construct_dict):
    """
    Train a model given a construction dictionairy
    """

    # Setup Log 
    if construct_dict["log_wandb"]:
        import wandb
        run = wandb.init(project = construct_dict["Exp_group"], entity = "johannbs", reinit = True, name = construct_dict["Experiment"], config = construct_dict)

    ################################################
    #   Load dataset for training and validation   #
    ################################################
    
    epochs      = construct_dict['epochs']
    batch_size  = construct_dict['batch_size']
    early_stop  = construct_dict['early_stop']
    patience    = construct_dict['patience']
    
    from scripts.datasets import graph_dataset

    train_data    = graph_dataset(construct_dict, "train", initialize = True)
    train_loader  = DisjointLoader(train_data, epochs = epochs, batch_size = batch_size)

    if construct_dict['clear_dataset']:
        construct_dict['clear_dataset'] = False


    val_data      = graph_dataset(construct_dict, "val")




    ################################################
    #   Setup, Loss_func and Train_loop            #
    ################################################

    # Get model, metrics, lr_schedule and loss function
    model, model_path     = setup_model(construct_dict)
    loss_func             = get_loss_func(construct_dict['LossFunc'])
    metrics               = get_metrics(construct_dict['metrics'])
    lr_schedule           = get_lr_schedule(construct_dict)

    # Learning rate and optimizer
    lr            = next(lr_schedule)
    opt           = Adam(lr)

    if construct_dict['angles_to_units']:
        from scripts.helper_functions import angles_to_units


    # Define training function
    @tf.function(input_signature = train_loader.tf_signature(), experimental_relax_shapes = True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            targets     = tf.cast(targets[:, 1:], tf.float32)
            if construct_dict['angles_to_units']:
                targets = angles_to_units(targets)
            loss        = loss_func(targets, predictions)
            loss       += sum(model.losses)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    
    # Define validation function
    @tf.function(input_signature = train_loader.tf_signature(), experimental_relax_shapes = True)
    def test_step(inputs, labels):
        ids, targets    = labels[:,0], labels[:, 1:]
        predictions     = model(inputs, training = False)
        targets         = tf.cast(targets, tf.float32) 
        if construct_dict['angles_to_units']:
            targets = angles_to_units(targets)
        return ids, targets, predictions


    def validation(loader, metrics):
        all_predictions = []
        all_targets     = []

        batches = 0
        loss    = 0

        # Loop over the batch and calculate predictions
        for batch in loader:
            inputs, labels  = batch
            _, targets, predictions = test_step(inputs, labels)
            loss           += loss_func(targets, predictions)
            batches        += 1
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        # Calculate validations
        val_loss    = loss / batches
        predictions = tf.concat(all_predictions, axis = 0)
        targets     = tf.concat(all_targets,     axis = 0)

        # Loop over metics and calculate
        if len(metrics) > 0:
            metric_values = [float(m(targets, predictions)) for m in metrics]
            metric_dict   = {i:j for i, j in zip(construct_dict['metrics'], metric_values)}
        else:
            metric_dict   = None

        return val_loss, metric_dict



    def test_model(loader, metrics):
        # Empty arrays
        all_predictions = []
        all_targets     = []
        all_Ns          = []
        all_ids         = []


        batches = 0
        loss    = 0

        # Loop over the batch and calculate predictions
        for batch in loader:
            inputs, labels  = batch
            _, __, N        = tf.unique_with_counts(inputs[2])
            ids, targets, predictions = test_step(inputs, labels)
            loss           += loss_func(targets, predictions)
            batches        += 1
            all_predictions.append(predictions)
            all_targets.append(targets)
            all_Ns.append(N)
            all_ids.append(ids)
        
        # Calculate validations
        test_loss   = loss / batches
        predictions = tf.concat(all_predictions, axis = 0)
        targets     = tf.concat(all_targets,     axis = 0)
        Ns          = tf.concat(all_Ns,          axis = 0)
        ids         = tf.concat(all_ids,         axis = 0)

        # Test metrics
        if len(metrics) > 0:
            metric_values = [float(m(targets, predictions)) for m in metrics]
            metric_dict   = {i:j for i, j in zip(construct_dict['metrics'], metric_values)}
        else:
            metric_dict   = {}
        
        metric_dict.update({"TestLoss": test_loss.numpy()})

        # Test dictionairy
        test_dict = {
            "id":       ids.numpy(),
            "reco":     predictions.numpy(),
            "targets":  targets.numpy(),
            "Ns":       Ns.numpy(),
            "metrics":  metric_dict
        }

        return test_dict

    ################################################
    #  Train Model                                 #      
    ################################################

    # Setup variables
    current_batch         = 0
    loss                  = 0
    validation_track_loss = []
    seen_data             = 0
    early_stop            = False
    batch_time            = np.ones(10) * 100
    clock                 = time.time()

    # Start loop
    while seen_data < construct_dict['train_data']:
        # Break if early_stopping is going
        if early_stop == True:
            break

        train_data    = graph_dataset(construct_dict, "train")
        train_loader  = DisjointLoader(train_data, epochs = 1, batch_size = batch_size)
        
        for batch in train_loader:
            # Train model
            inputs, labels  = batch
            out             = train_step(inputs, labels)
            loss           += out

            # Update counters
            seen_data      += len(labels)
            current_batch  += 1
            batch_time      = np.roll(batch_time, 1)
            batch_time[0]   = time.time() - clock
            clock           = time.time()
            
            # Print if verbose
            if construct_dict['verbose']: 
                print(f"Seen data: {seen_data:07d} \t Avg loss since last validation: {loss / current_batch:.6f} \t Data per second: {construct_dict['batch_size'] / np.mean(batch_time):.2f}          ", end = "\r")

            
            # Validate and update learning rate
            if seen_data % construct_dict['val_every'] < batch_size and seen_data > 0:

                # Validate data
                val_loader = DisjointLoader(val_data, epochs = 1, batch_size = batch_size)
                val_loss, metric_dict = validation(val_loader, metrics)

                # Print if verbose
                if construct_dict['verbose']:
                    print("\n")
                    print(f"Validation loss: {val_loss :.6f} \t learning_rate {lr:.3e}")
                    print(f"Validation metrics: " + "\t".join([f"{i}: {metric_dict[i]:.3f}" for i in metric_dict.keys()]))
                    print("")


                # Log to wandb
                if construct_dict["log_wandb"]:
                    to_log              = {}
                    if len(metric_dict) > 0:
                        for m in metric_dict.keys():
                            to_log[m] = metric_dict[m]

                    to_log["train_loss"]    = loss / current_batch 
                    to_log["val_loss"]      = val_loss
                    to_log["learning_rate"] = lr

                    wandb.log(data = to_log, step = seen_data)

                # Update learning rate according to schedule
                lr  = next(lr_schedule)
                opt.learning_rate.assign(lr)
                
                # Reset states
                current_batch    = 0
                loss             = 0

                # Save model
                model.save(model_path)

                # Check for early_stopping
                validation_track_loss.append(val_loss)

                if (np.argmin(validation_track_loss) < len(validation_track_loss) - patience) and construct_dict['early_stop'] == True:
                    if construct_dict['verbose']:
                        print(f"Training stopped, no improvement made in {patience} steps.")
                        early_stop = True
                        break

    test_data             = graph_dataset(construct_dict, "test")
    test_loader           = DisjointLoader(test_data, epochs = 1, batch_size = batch_size, shuffle = False)
    test_dict             = test_model(test_loader, metrics)

    if construct_dict["log_wandb"]:
        wandb.log(test_dict['metrics'])
        run.finish()

    with open(osp.join(file_path, "..", "test_folder", construct_dict['Experiment'] + ".dat"), "wb") as file:
        pickle.dump(test_dict, file)

    print(model.summary())

    



def get_lr_schedule(construct_dict):
    lr        = construct_dict['learning_rate']
    schedule  = construct_dict['lr_schedule']

    import scripts.lr_schedules as lr_module

    lr_generator = getattr(lr_module, schedule)

    lr_schedule  = lr_generator(lr)()

    return lr_schedule



def get_metrics(metric_names):
    # Returns a list of functions
    metric_list = []
    import scripts.metrics as metrics
        
    for name in metric_names:
        metric_list.append(getattr(metrics, name))

    return metric_list


def get_loss_func(name):
    # Return loss func from the loss functions folder given a name
    import scripts.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    return loss_func



def setup_model(construct_dict):
    # Retrieve name and params for construction
    model_name    = construct_dict['ModelName']
    hyper_params  = construct_dict['hyper_params']
    experiment    = construct_dict['Experiment']

    # Load model from model folder
    if construct_dict["Load_model"]:
        model = tf.keras.models.load_model(osp.join(file_path, "..", "models", construct_dict["Load_name"]))
    else:
        import scripts.models as model_module
        model         = getattr(model_module, model_name) 
        model         = model(**hyper_params)

        # Make folder for saved states
    model_path    = osp.join(file_path, "..", "models", experiment)
    if not osp.isdir(model_path):
        os.mkdir(model_path)

    return model, model_path

