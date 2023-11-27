# This file contains tests for the filemanagement.py module

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from filemanagement import load_data, count_files_in_folder

# Mock data for testing
SAMPLE_RATE = 44100
N_MFCC = 26
NORMALIZE_LENGTH = 5  # seconds

# Create a mock audio file (a numpy array simulating audio data)
mock_audio_data = np.random.rand(SAMPLE_RATE * NORMALIZE_LENGTH)


# Mock librosa.load function to return the mock audio data
def mock_load(file_path, sr=SAMPLE_RATE):
    return mock_audio_data, sr


# Mock os.listdir to control the list of files returned
@patch('librosa.load', side_effect=mock_load)
@patch('os.listdir')
def test_load_data(mock_listdir, mock_load):  # Notice the order and inclusion of mock objects as arguments
    """
    Test the load_data function from filemanagement.py
    """
    # Setup mock environment
    mock_listdir.return_value = ['file1.wav', 'file2.wav']

    # Define expected features and labels
    expected_feature_length = N_MFCC
    expected_label = 1
    expected_number_of_files = 2

    # Call the function under test
    features, labels = load_data('mock_folder', expected_label, SAMPLE_RATE, N_MFCC, NORMALIZE_LENGTH)

    # Assertions
    assert len(features) == expected_number_of_files, "Incorrect number of features returned"
    assert len(labels) == expected_number_of_files, "Incorrect number of labels returned"
    assert all(len(feature) == expected_feature_length for feature in features), "Feature length mismatch"
    assert all(label == expected_label for label in labels), "Label mismatch"


# Test count_files_in_folder function
@patch('os.path.exists', return_value=True)
@patch('os.listdir', return_value=['file1.txt', 'file2.txt'])
@patch('os.path.isfile')
def test_count_files_in_folder(mock_isfile, mock_listdir, mock_path_exists):
    """
    Test the count_files_in_folder function from filemanagement.py
    """
    # Setup mock environment to consider all entries in listdir as files
    mock_isfile.return_value = True

    # Call the function under test
    count = count_files_in_folder('mock_folder')

    # Assertions
    assert count == 2, "File count should be 2"


# Test count_files_in_folder function for non-existing folder
@patch('os.path.exists')
def test_count_files_in_non_existing_folder(mock_path_exists):
    """
    Test count_files_in_folder function for a non-existing folder
    """
    # Setup mock environment
    mock_path_exists.return_value = False

    # Call the function under test
    response = count_files_in_folder('non_existing_folder')

    # Assertions
    assert response == "Folder not found.", "Should return 'Folder not found.' for non-existing folder"
