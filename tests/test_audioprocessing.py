# Tests for the audioprocessing module

import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
import torch

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
    # Mock model with necessary attributes and methods
    model = MagicMock()
    model.scaler = MagicMock()
    model.eval = Mock()

    # Mock model call to return a tensor-like object
    mock_output_tensor = torch.tensor([[0, 1]], dtype=torch.float32)  # Simulate a tensor returned by the model
    model.return_value = mock_output_tensor  # When model is called, it returns the mock tensor

    return model


@pytest.fixture
def mock_data_plotter():
    # Mock data plotter with necessary methods
    plotter = MagicMock()
    plotter.update_mfcc_data = Mock()
    plotter.update_prediction_text = Mock()
    plotter.animate = Mock()
    return plotter


def test_audio_callback(mock_model, mock_data_plotter):
    """
    Test the audio_callback method of the AudioProcessor class.
    """
    # Create an instance of AudioProcessor with mock dependencies
    audio_processor = AudioProcessor(mock_model, mock_data_plotter, SAMPLE_RATE, BLOCK_SIZE, N_MFCC, N_FFT, HOP_LENGTH, verbose=True)

    # Mock the status object passed to the callback
    mock_status = Mock()

    # Call the audio_callback method
    audio_processor.audio_callback(mock_audio_buffer, BLOCK_SIZE, None, mock_status)

    # Assertions to verify the behavior
    mock_model.eval.assert_called_once()  # Check if model's eval method was called
    mock_model.scaler.transform.assert_called()  # Check if scaler's transform method was called
    mock_data_plotter.update_mfcc_data.assert_called()  # Check if plotter's update_mfcc_data was called
    mock_data_plotter.update_prediction_text.assert_called()  # Check if plotter's update_prediction_text was called

    # Assert the detected_count is updated correctly
    assert audio_processor.detected_count == 0 or audio_processor.detected_count == 1


def test_compute_mfcc():
    """
    Test the compute_mfcc function.
    """
    mfccs = compute_mfcc(mock_audio_buffer, SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH)
    assert mfccs.shape[0] == N_MFCC, "MFCCs should have the correct number of coefficients"


def test_normalize_audio_length():
    """
    Test the normalize_audio_length function.
    """
    # Setup: Create a mock audio file shorter than the target length
    short_audio_length = 3  # seconds (shorter than target)
    short_audio_data = np.random.rand(short_audio_length * SAMPLE_RATE)

    # Setup: Create a mock audio file longer than the target length
    long_audio_length = 10  # seconds (longer than target)
    long_audio_data = np.random.rand(long_audio_length * SAMPLE_RATE)

    # Target length for normalization
    target_length = AUDIO_LENGTH  # 7 seconds

    # Mock librosa.load to return short and long audio data
    with patch('librosa.load', side_effect=[(short_audio_data, SAMPLE_RATE), (long_audio_data, SAMPLE_RATE)]):
        # Normalize short audio
        normalized_short_audio = normalize_audio_length('path/to/short_audio.wav', target_length, SAMPLE_RATE)
        # Normalize long audio
        normalized_long_audio = normalize_audio_length('path/to/long_audio.wav', target_length, SAMPLE_RATE)

    # Assertions
    assert len(normalized_short_audio) == target_length * SAMPLE_RATE, "Short audio should be padded to target length"
    assert len(normalized_long_audio) == target_length * SAMPLE_RATE, "Long audio should be truncated to target length"

