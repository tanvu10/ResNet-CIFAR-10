from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
import torch

import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE 18
    parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=2, 
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=16, help='number of classes')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    parser.add_argument("--lr_decay_step", type=int, default=10, help='number of step (epoch) to decay lr')
    parser.add_argument("--learning_rate", type=int, default=0.01, help='learning rate')

    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    # data_dir = "/Users/vuh/Downloads/cifar-10-batches-py/"
    data_dir = '/gpfs/scratch/vhuynh/cifar-10-batches-py/'

    ### YOUR CODE HERE
    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    # model = Cifar(config).cuda()
    model = Cifar(config)
    model = model.to(model.device)
    print(f'detected {model.device}, using {model.device}')

    ### YOUR CODE HERE
    # First step: use the train_new set and the valid set to choose hyperparameters.
    model.train(x_train_new, y_train_new, 200)
    model.test_or_validate(x_valid, y_valid, [160, 170, 180, 190, 200])

    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    # model.train(x_train, y_train, 10)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # model.test_or_validate(x_test, y_test, [10])
    ### END CODE HERE

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)