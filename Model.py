import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet
from ImageUtils import parse_record

""" This script defines the training, validation and testing process.
"""

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.network = ResNet(
            self.config.resnet_version,
            self.config.resnet_size,
            self.config.num_classes,
            self.config.first_num_filters,
        )
        ### YOUR CODE HERE
        # define cross entropy loss and optimizer
        self.criterion = nn.CrossEntropyLoss()

        # add L2 weight decay in the optimizer
        self.optimizer = torch.optim.SGD(self.network.parameters(), 
                                         lr=self.config.learning_rate, 
                                         weight_decay=self.config.weight_decay)
        
        # add feature lr_decay by  = 1/10 every lr_decay_step
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=self.config.lr_decay_step, 
                                                         gamma=0.1)

        ### YOUR CODE HERE
    
    def train(self, x_train, y_train, max_epoch):
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // self.config.batch_size

        print('### Training... ###')
        for epoch in range(1, max_epoch+1):
            start_time = time.time()
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            ### YOUR CODE HERE
            # Set the learning rate for this epoch
            # Usage example: divide the initial learning rate by 10 after several epochs
            self.scheduler.step()
            ### YOUR CODE HERE
            
            for i in range(num_batches):
                ### YOUR CODE HERE
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                # Don't forget L2 weight decay
                start = i * self.config.batch_size
                end = start + self.config.batch_size

                x_batch = np.array([parse_record(image, training=True) for image in curr_x_train[start:end]])
                y_batch = curr_y_train[start:end]

                # change to tensor type
                x_batch = torch.tensor(x_batch, dtype=torch.float32)
                y_batch = torch.tensor(y_batch, dtype=torch.int32)

                outputs = self.network(x_batch)
                loss = self.criterion(outputs, y_batch)
                
                ### YOUR CODE HERE
                # reset gradients for each batch
                self.optimizer.zero_grad()
                # back propagation
                loss.backward()
                # update params
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, loss, duration))

            # save all params every save_interval_epoch
            if epoch % self.config.save_interval == 0:
                self.save(epoch)


    def test_or_validate(self, x, y, checkpoint_num_list):
        # change to test mode (since using BN params from training process)
        self.network.eval()
        print('### Test or Validation ###')
        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(checkpoint_num))
            self.load(checkpoint_file)

            preds = []
            for i in tqdm(range(x.shape[0])):
                ### YOUR CODE HERE
                # avoid calculating gradient
                with torch.no_grad():
                    image = parse_record(x[i], training=False)
                    # change dimension from [H, W, C] to [1, H, W, C]
                    image = image.squeeze(0)
                    image = torch.tensor(image, dtype=torch.float32)

                    output = self.network(image)
                    prob = nn.functional.softmax(output, dim=1)
                    pred_class = torch.argmax(prob, dim=1)
                    # take out scalar value
                    preds.append(pred_class.item())
                ### END CODE HERE

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print('Test accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]))
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config.modeldir, 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config.modeldir, exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))