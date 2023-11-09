from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
import torch
import json
import os
import argparse
import torch.nn as nn
import sys
from log import DualLogger

# def configure():
#     parser = argparse.ArgumentParser()
#     ### YOUR CODE HERE 18
#     parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
#     parser.add_argument("--resnet_size", type=int, default=2, 
#                         help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
#     parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
#     parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
#     parser.add_argument("--save_interval", type=int, default=10, 
#                         help='save the checkpoint when epoch MOD save_interval == 0')
#     parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
#     parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
#     parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
#     parser.add_argument("--lr_decay_step", type=int, default=10, help='number of step (epoch) to decay lr')
#     parser.add_argument("--learning_rate", type=int, default=0.01, help='learning rate')

#     ### YOUR CODE HERE
#     return parser.parse_args()


class Config:
    def __init__(self):
        self.resnet_version = 1
        self.resnet_size = 18
        self.batch_size = 128
        self.num_classes = 10
        self.save_interval = 10
        self.first_num_filters = 16
        self.weight_decay = 1e-4
        self.modeldir = 'model_v1'
        self.lr_decay_step = 10
        self.learning_rate = 0.01

def tune_hyperparameters(hyperparameter_space, x_train_new, y_train_new, x_valid, y_valid):
    # data_dir = '/gpfs/scratch/vhuynh/cifar-10-batches-py/'
    # x_train, y_train, x_test, y_test = load_data(data_dir)
    # x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    i = 1
    best_accuracy = 0
    all_hyperparams_info = []
    best_hyperparams = {}

    current_directory = os.getcwd()
    valid_model_path = current_directory + '/resnet_paper'


    for resnet_version in hyperparameter_space['resnet_version']:
        for resnet_size in hyperparameter_space['resnet_size']:
            for batch_size in hyperparameter_space['batch_size']:

                config = Config()
                config.resnet_version = resnet_version
                config.resnet_size = resnet_size
                config.batch_size = batch_size
                if resnet_version == 1:
                    config.modeldir = valid_model_path + f'/resnet_110_{batch_size}'
                else:
                    config.modeldir = valid_model_path + f'/resnet_164_{batch_size}'
                    
                model = Cifar(config)
                model = model.to(model.device)

                print(f'detected {model.device}, now using {model.device}')
                model.train(x_train_new, y_train_new, 30)
                best_epoch = None
                sub_best_accuracy = 0
                
                accuracy = model.test_or_validate(x_valid, y_valid, [10, 20, 30])

                for idx, acc in enumerate(accuracy):
                    acc_value = acc.item()  # Converts the single-element tensor to a float
                    if acc_value > sub_best_accuracy:
                        sub_best_accuracy = acc_value
                        best_epoch = (idx + 1) * 10
                

                # save for reporting
                hyperparams_info = {
                    'model_valid_ver': i,
                    'resnet_version': resnet_version,
                    'resnet_size': resnet_size,
                    'batch_size': batch_size,
                    'accuracy': sub_best_accuracy,
                    'epoch': best_epoch
                }

                all_hyperparams_info.append(hyperparams_info)


                if sub_best_accuracy > best_accuracy:
                    best_accuracy = sub_best_accuracy
                    best_hyperparams = hyperparams_info

                i += 1


    # Define the file path
    best_param_path = os.path.join(current_directory, 'best_hyperparams_paper.json')
    all_hyperparams_info_path = os.path.join(current_directory, 'all_hyperparams_info_paper.json')

    # Save to JSON file in the current directory
    with open(best_param_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    
    with open(all_hyperparams_info_path, 'w') as f:
        json.dump(all_hyperparams_info, f, indent=4)
                
    return best_hyperparams
    

def main():

    log_file_path = "resnet_paper_log.txt"
    # Start logging to both terminal and file
    with DualLogger(log_file_path) as logger:
        sys.stdout = logger

        print("--- Preparing Data ---")

        ### YOUR CODE HERE
        # data_dir = "/Users/vuh/Downloads/cifar-10-batches-py/"
        data_dir = '/gpfs/scratch/vhuynh/cifar-10-batches-py/'

        ### YOUR CODE HERE
        x_train, y_train, x_test, y_test = load_data(data_dir)
        x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

        # model = Cifar(config).cuda()
        
        ### YOUR CODE HERE
        # First step: use the train_new set and the valid set to choose hyperparameters.
        # model.train(x_train_new, y_train_new, 200)
        # model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])

        hyperparameter_space = {
        'batch_size': [128, 256],
        'resnet_size': [18],
        'resnet_version': [1, 2]
        }
        best_hyperparams = tune_hyperparameters(hyperparameter_space, x_train_new, y_train_new, x_valid, y_valid)

        # Second step: with hyperparameters determined in the first run, re-train
        # your model on the original train set.
        current_directory = os.getcwd()

        config = Config()
        config.resnet_version = best_hyperparams['resnet_version']
        config.resnet_size = best_hyperparams['resnet_size']
        config.batch_size = best_hyperparams['batch_size']
        config.modeldir = current_directory + f'/final_resnet_paper'
        model = Cifar(config)
        model = model.to(model.device)

        print(f'detected {model.device}, now using {model.device}')
        # train on full train set
        model.train(x_train, y_train, 30)

        # Third step: after re-training, test your model on the test set.
        # Report testing accuracy in your hard-copy report.
        final_accuracy = model.test_or_validate(x_test, y_test, [best_hyperparams['epoch']])[0]
        print(f'final accuracy is {final_accuracy}')
        ### END CODE HERE

        
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
    