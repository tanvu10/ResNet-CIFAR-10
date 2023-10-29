import numpy as np
""" This script implements the functions for data augmentation and preprocessing.
"""

def parse_record(record, training):
    """ Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def preprocess_image(image, training):
    """ Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    if training:
        ### YOUR CODE HERE
        # Resize the image to add four extra pixels on each side.
        # 4 pad before 4 pad after for height and width
        # no padding for 3rd dimension
        image = np.pad(image, [(4, 4), (4, 4), (0, 0)], mode='constant')
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE
        # Randomly crop a [32, 32] section of the image.
        # HINT: randomly generate the upper left point of the image
        start_x = np.random.randint(0, image.shape[0] - 32)
        start_y = np.random.randint(0, image.shape[1] - 32)
        image = image[start_x:start_x+32, start_y:start_y+32, :]
        ### YOUR CODE HERE
        
        ### YOUR CODE HERE
        # Randomly flip the image horizontally.
        # flip with pro = 0.5
        if np.random.rand() <= 0.5:
            image = np.fliplr(image)
        ### YOUR CODE HERE
        
        
    ### YOUR CODE HERE
    # Subtract off the mean and divide by the standard deviation of the pixels.
    # Channel-wise normalization
    for i in range(3): 
        mean = np.mean(image[:, :, i])
        std = np.std(image[:, :, i])
        image[:, :, i] = (image[:, :, i] - mean) / std
    ### YOUR CODE HERE
    return image

if __name__ == '__main__':
    pass