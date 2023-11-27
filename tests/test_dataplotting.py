# Tests for the dataplotting module

import pytest
import numpy as np
from dataplotting import AudioDataPlotter

# Constants for testing
N_MFCC = 26
MFCC_RANGE = 10

# Mock MFCC data for testing
mock_mfcc_data = np.random.rand(N_MFCC)


def test_audio_data_plotter_initialization():
    """
    Test initialization of the AudioDataPlotter class.
    """
    plotter = AudioDataPlotter(N_MFCC, MFCC_RANGE)

    # Assertions to verify initialization
    assert plotter.n_mfcc == N_MFCC, "Incorrect number of MFCC coefficients"
    assert plotter.mfcc_range == MFCC_RANGE, "Incorrect MFCC range"
    assert plotter.mfccs.shape[0] == N_MFCC, "MFCCs should be initialized to zeros with correct shape"


def test_update_mfcc_data():
    """
    Test the update_mfcc_data method of the AudioDataPlotter class.
    """
    plotter = AudioDataPlotter(N_MFCC, MFCC_RANGE)

    # Update the MFCC data
    plotter.update_mfcc_data(mock_mfcc_data)

    # Assertions to verify the update
    assert np.array_equal(plotter.mfccs, mock_mfcc_data), "MFCC data should be updated correctly"


def test_update_prediction_text():
    """
    Test the update_prediction_text method of the AudioDataPlotter class.
    """
    plotter = AudioDataPlotter(N_MFCC, MFCC_RANGE)
    test_text = "Test Prediction"

    # Update the prediction text
    plotter.update_prediction_text(test_text)

    # Assertions to verify the update
    assert plotter.prediction_text == test_text, "Prediction text should be updated correctly"


def test_update_filter_activity_text():
    """
    Test the update_filter_activity_text method of the AudioDataPlotter class.
    """
    plotter = AudioDataPlotter(N_MFCC, MFCC_RANGE)
    test_text = "Filter Active"
    test_transparency = 0.5

    # Update the filter activity text
    plotter.update_filter_activity_text(test_text, test_transparency)

    # Assertions to verify the update
    assert plotter.filter_activity_text == test_text, "Filter activity text should be updated correctly"
    assert plotter.filter_activity_text_element.get_alpha() == test_transparency, "Filter activity text transparency should be updated correctly"
