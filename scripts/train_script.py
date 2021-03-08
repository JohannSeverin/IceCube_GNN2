import tqdm, os, sys

import tensorflow as tf
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
        run = wandb.init(project = construct_dict["Exp_group"], entity = "johannbs", reinit = True, name = "Experiment")
        run.config(construct_dict)

    ################################################
    #   Load dataset for training and validation   #
    ################################################
    
    epochs      = construct_dict['epochs']
    batch_size  = construct_dict['batch_size']
    early_stop  = construct_dict['early_stop']
    patience    = construct_dict['patience']
    
    from scripts.datasets import graph_dataset

    train_data    = graph_dataset(construct_dict, "train")
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

    # Define training function
    @tf.function(input_signature = train_loader.tf_signature(), experimental_relax_shapes = True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            targets     = tf.cast(targets, tf.float32)
            loss        = loss_func(targets, predictions)
            loss       += sum(model.losses)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    
    # Define validation function
    @tf.function(input_signature = train_loader.tf_signature(), experimental_relax_shapes = True)
    def validation(loader, metrics):
        all_predictions = []
        all_targets     = []

        batches = 0

        # Loop over the batch and calculate predictions
        for batch in loader:
            inputs, targets = batch
            predictions     = model(inputs, training = False)
            targets         = tf.cast(targets, tf.float32) 
            loss           += loss_func(targets, predictions)
            batches        += 1
            all_predictions.append(predictions)
            all_targets.append(targets)
        
        # Calculate validations
        val_loss    = loss / batches
        predictions = tf.concat(all_predictions, axis = 1)
        targets     = tf.concat(all_targets,     axos = 1)

        # Loop over metics and calculate
        if len(metrics) > 0:
            metric_values = [m(targets, predictions) for m in metrics]
            metric_dict   = {i:j for i, j in zip(construct_dict['metrics'], metric_values)}
        else:
            metric_dict   = None

        return val_loss, metric_dict



    ################################################
    #  Train Model                                 #      
    ################################################

    # Setup variables
    current_batch         = 0
    loss                  = 0
    validation_track_loss = []
    seen_data             = 0     


    # Start loop
    for batch in train_loader:

        # Train model
        inputs, targets = batch
        out             = train_step(inputs, targets)
        loss           += out

        # Update counters
        seen_data      += len(targets)
        current_batch  += 1
        
        # Print if verbose
        if construct_dict['verbose']: 
            print(f"Seen data: {seen_data} \t Avg loss since last validation: {loss / current_batch:.6f}          ", end = "\r")

        
        # Validate and update learning rate
        if seen_data % construct_dict['val_every'] < batch_size and seen_data > 0:

            # Validate data
            val_loader = DisjointLoader(val_data, epoch = 1, batch_size = batch_size)
            val_loss, metric_dict = validation(val_loader)

            # Print if verbose
            if construct_dict['verbose']:
                print(f"Validation loss: {val_loss :.6f}")
                print(f"Validation metrics: " + "\t".join([f"{i}: {metric_dict[i]:.6f}" for i in metric_dict.keys()]))

            # Log to wandb
            if construct_dict["log_wandb"]:
                to_log              = {}
                if len(metric_dict) > 0:
                    for m in metric_dict.keys():
                        to_log[m] = metric_dict[m]

                to_log["train_loss"]    = loss / current_batch 
                to_log["val_loss"]      = val_loss
                to_log["learning_rate"] = lr

                run.log(data = to_log, step = seen_data)

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

            if np.argmin(validation_track_loss) > len(validation_track_loss) - patience and early_stop == True:
                if construct_dict['verbose']:
                    print(f"Training stopped, no improvement made in {patience} steps.")
    
    run.finish()
    



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

    # Load model from model folder
    import scripts.models as model_module
    model         = getattr(model_module, model_name) 
    model         = model(**hyper_params)

    # Make folder for saved states
    model_path    = osp.join(file_path, "..", "models", model_name)
    if not osp.isdir(model_path):
        os.mkdir(model_path)

    return model, model_path

