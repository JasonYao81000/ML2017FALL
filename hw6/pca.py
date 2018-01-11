# bash pca.sh ../imgs 414.jpg
# python3 pca.py ../imgs 414.jpg

import os
import sys
import numpy as np
from skimage import io

RECONSTRUCT_DIM = 4
EIGENFACES_NUMBER = 10

if __name__ == "__main__":
    np.random.seed(777)
    print ('Fixed random seed for reproducibility.')

    # Load path of image folder from arguments.
    IMAGE_FOLDER_PATH = sys.argv[1]
    # Load file name of image from arguments.
    IMAGE_FILE_NAME = sys.argv[2]
    
    # Load file name of images.
    image_file_list = os.listdir(IMAGE_FOLDER_PATH)

    # Sample an image.
    sample_image = io.imread(os.path.join(IMAGE_FOLDER_PATH, IMAGE_FILE_NAME))
    # print ('sample_image.shape: ', sample_image.shape)
    # # Save sample image to image file.
    # io.imsave('sample_image.jpg', sample_image)

    # Allocate memory space of images.
    images = np.zeros((len(image_file_list), sample_image.shape[0], sample_image.shape[1], sample_image.shape[2]))
    # print ('images.shape: ', images.shape)

    print ('Loading images to numpy array...')
    for index, filename in enumerate(image_file_list):
        images[index] = io.imread(os.path.join(IMAGE_FOLDER_PATH, filename))
    
    # Mean of all images.
    mean_images = np.mean(images, axis = 0)
    # print ('mean_images.shape: ', mean_images.shape)
    # # Save Mean to image file.
    # io.imsave('mean.jpg', mean_images.astype(np.uint8))

    # Reshape each image to 1-dimension.
    images = images.reshape(images.shape[0], -1)
    # print ('images.shape: ', images.shape)
    mean_images = mean_images.reshape(-1)
    # print ('mean_images.shape: ', mean_images.shape)

    print ('Performing SVD on images...')
    # Remember to transpose before SVD.
    U, s, V = np.linalg.svd((images - mean_images).T, full_matrices=False)
    # # Save U, s, V to npy files.
    # np.save('U.npy', U)
    # np.save('s.npy', s)
    # np.save('V.npy', V)

    # # Load U, s, V to from files.
    # U = np.load('U.npy')
    # s = np.load('s.npy')
    # V = np.load('V.npy')
    # print ('U.shape: ', U.shape)
    # print ('s.shape: ', s.shape)
    # print ('V.shape: ', V.shape)

    # print ('Fetching eigenfaces from U...')
    # eigenfaces = -1 * U[:, 0:EIGENFACES_NUMBER].T
    # # Rescale to 0~255.
    # eigenfaces -= np.min(eigenfaces)
    # eigenfaces /= np.max(eigenfaces)
    # eigenfaces = (eigenfaces * 255).astype(np.uint8)
    # print ('eigenfaces.shape: ', eigenfaces.shape)
    # # Reshape each eigenfaces to RGB format.
    # eigenfaces = eigenfaces.reshape(eigenfaces.shape[0], sample_image.shape[0], sample_image.shape[1], sample_image.shape[2])
    # print ('eigenfaces.shape: ', eigenfaces.shape)
    # print ('Saving eigenfaces to image file...')
    # for i in range(EIGENFACES_NUMBER):
    #     io.imsave('eigenface' + str(i) + '.jpg', eigenfaces[i].astype(np.uint8))
    
    # # Calculate proportion of each s.
    # for i in range(RECONSTRUCT_DIM):
    #     proportion = s[i] / sum(s) * 100
    #     print ('s[%d] = %f, sum(s) = %f' % (i, s[i], sum(s)))
    #     print ('s[%d] = %f %%' % (i, proportion))
    #     print ('s[%d] = %0.1f %%' % (i, proportion))

    print ('Reconstructing image...')
    weights = np.dot(images - mean_images, U)
    M = np.dot(weights[:, :RECONSTRUCT_DIM], U[:, :RECONSTRUCT_DIM].T) + mean_images
    # print ('M.shape: ', M.shape)
    # Rescale to 0~255.
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    # Reshape each image to RGB format.
    M = M.reshape(M.shape[0], sample_image.shape[0], sample_image.shape[1], sample_image.shape[2])
    # print ('M.shape: ', M.shape)
    # Save sample image to image file.
    # print ('IMAGE_FILE_NAME: ', IMAGE_FILE_NAME)
    # print ('image_file_list.index(IMAGE_FILE_NAME): ', image_file_list.index(IMAGE_FILE_NAME))
    io.imsave('reconstruction.jpg', M[image_file_list.index(IMAGE_FILE_NAME)].astype(np.uint8))