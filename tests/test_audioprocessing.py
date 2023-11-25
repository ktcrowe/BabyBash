# Tests for the audioprocessing module

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock, Mock

# Import the AudioProcessor class
from audioprocessing import AudioProcessor, compute_mfcc, normalize_audio_length

# Mock constants for testing
SAMPLE_RATE = 44100
N_MFCC = 26
BLOCK_SIZE = 882
N_FFT = BLOCK_SIZE
HOP_LENGTH = N_FFT // 4
AUDIO_LENGTH = 7  # seconds

# Create a mock audio buffer (numpy array simulating audio data)
mock_audio_buffer = np.random.rand(BLOCK_SIZE, 1)


@pytest.fixture
def mock_model():
    """
    Fixture to create a mock model with necessary attributes and methods.
    """
    model = MagicMock()
    model.eval = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9]])  # Simulate prediction tensor
    return model


@pytest.fixture
def mock_data_plotter():
    """
    Fixture to create a mock data plotter with necessary methods.
    """
    plotter = MagicMock()
    plotter.update_mfcc_data = MagicMock()
    plotter.update_prediction_text = MagicMock()
    plotter.animate = MagicMock()
    return plotter


def test_audio_callback(mock_model, mock_data_plotter):
    """
    Test the audio_callback method in the AudioProcessor class.
    """
    processor = AudioProcessor(mock_model, mock_data_plotter, SAMPLE_RATE, BLOCK_SIZE, N_MFCC, N_FFT, HOP_LENGTH, 1, 2,
                               verbose=True)

    # Mock input and output data for callback
    indata = np.random.rand(BLOCK_SIZE, 1)
    outdata = np.zeros_like(indata)

    # Call the audio_callback method
    processor.audio_callback(indata, outdata, BLOCK_SIZE, None, None)

    # Verify that the model's eval method was called
    mock_model.eval.assert_called_once()

    # Verify update methods on data plotter were called
    mock_data_plotter.update_mfcc_data.assert_called()
    mock_data_plotter.update_prediction_text.assert_called()

    # Verify changes in output data
    assert not np.array_equal(outdata, np.zeros_like(indata)), "Output data should be modified in the callback"


def test_compute_mfcc():
    """
    Test the compute_mfcc function to ensure it returns MFCCs with the correct shape.
    """
    mfccs = compute_mfcc(mock_audio_buffer, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH)
    assert mfccs.shape == (N_MFCC,), "MFCCs should have the correct number of coefficients"


# Updated test_normalize_audio_length
def test_normalize_audio_length():
    """
    Test the normalize_audio_length function to ensure it correctly pads or truncates audio data.
    """
    # Mocked audio data lengths
    short_audio_length = int(AUDIO_LENGTH / 2 * SAMPLE_RATE)
    long_audio_length = int(AUDIO_LENGTH * 1.5 * SAMPLE_RATE)

    # Mock librosa.load to return audio data of specific lengths
    with patch('librosa.load') as mock_load:
        # Configure the mock to return an array of specific length and the sample rate
        mock_load.side_effect = lambda file_path, sr: (np.zeros(short_audio_length), sr) if 'short' in file_path else (
        np.zeros(long_audio_length), sr)

        # Normalize short and long audio data
        normalized_short_audio = normalize_audio_length('path/to/short_audio.wav', AUDIO_LENGTH, SAMPLE_RATE)
        normalized_long_audio = normalize_audio_length('path/to/long_audio.wav', AUDIO_LENGTH, SAMPLE_RATE)

    # Assertions
    assert len(normalized_short_audio) == AUDIO_LENGTH * SAMPLE_RATE, "Short audio should be padded to target length"
    assert len(normalized_long_audio) == AUDIO_LENGTH * SAMPLE_RATE, "Long audio should be truncated to target length"
