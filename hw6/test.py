# bash hw6.sh ./image.npy ./test_case.csv ./prediction.csv
# python3 test.py ./image.npy ./prediction.csv ./test_case.csv ./labels.npy

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Fix random seed for reproducibility.
    seed = 777
    np.random.seed(seed)
    print ('Fixed random seed for reproducibility.')

    # Load file path of image.npy from arguments.
    IMAGE_NPY_FILE_PATH = sys.argv[1]
    # Load file path of prediction.csv from arguments.
    PREDICTION_CSV_FILE_PATH = sys.argv[2]
    # Load file path of test_case.csv from arguments.
    TEST_CSV_FILE_PATH = sys.argv[3]
    # Load file path of labels.npy from arguments.
    LABELS_NPY_FILE_PATH = sys.argv[4]

    # Read test dataframe from csv file.
    test_df = pd.read_csv(TEST_CSV_FILE_PATH)
    
    # Prepare id, image1_index, image2_index for testing.
    id = test_df['ID']
    image1_index = test_df['image1_index']
    image2_index = test_df['image2_index']
    # print (id[0])
    # print (image1_index[0])
    # print (image2_index[0])

    # Read image from npy file.
    print ('Reading image from npy file...')
    image = np.load(IMAGE_NPY_FILE_PATH)
    print ('image.shape: ', image.shape)
    # Read labels from npy file.
    labels = np.load(LABELS_NPY_FILE_PATH)
    print ('labels.shape: ', labels.shape)

    # Empty list of answers.
    answers = []
    # Check both of images' label.
    for n in range(len(id)):
    # for n in range(10):
        # Fetch labels by image index.
        label1 = labels[image1_index[n]]
        label2 = labels[image2_index[n]]
        # If two image are from the same cluster.
        if (label1 == label2):
            # '1' for the same cluster.
            answers.append('1')
        else:
            # '0' for the different cluster.
            answers.append('0')
        # Show progress each 0.1%.
        if (n % (len(id) / 100) == 0):
            print ('Cluster Processing: %02.00f%%' % (n * 100 / len(id)))

    prediction_df = pd.DataFrame(
        {'ID': id, 'Ans': answers}, 
        columns=('ID', 'Ans'))
    prediction_df.to_csv(
        PREDICTION_CSV_FILE_PATH, 
        index=False, 
        columns=('ID', 'Ans'))