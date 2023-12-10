# Import required libraries
import os
import sys

import gui
import audioclassification as ac  # Neural network for audio classification
import dataplotting as dp  # Tools for plotting and visualization of audio data
from filemanagement import load_data  # Tools for file management
from audioprocessing import AudioProcessor  # Tools for audio processing

# Define constant values for audio processing
SAMPLE_RATE = 44100  # Sample rate in Hz for audio processing
BLOCK_SIZE = 882  # Block size for the audio stream buffer (882 creates 20ms blocks)
NUM_MFCC_COEFFS = 26  # Number of Mel-frequency cepstral coefficients (MFCCs) to compute
MFCC_RANGE = 10  # Range of values for the y-axis on the plot (initially was 275, but reduced to reflect normalized MFCCs)
N_FFT = BLOCK_SIZE  # Fast Fourier Transform (FFT) window size set to block size
HOP_LENGTH = N_FFT // 4  # Hop length for the FFT, typically 1/4th of the FFT window size
AUDIO_FILE_LENGTH = 7  # Length of audio files in seconds (used for normalizing audio length)

# Define paths
CRYING_PATH = 'data/crying'  # audio files of crying babies
NOISE_PATH = 'data/noise'  # audio files of various other noises (not babies crying)
MODEL_TO_USE = 'model_v1.pth'  # the model to use for detecting crying babies (if USE_SPECIFIC_MODEL is True)
"""
The current models are as follows:
    - model_v1.pth: No file length normalization results are somewhere between v2 and v3 (this is the model used in alpha build)
    - model_v3.pth: File length normalization to 5s results in worse detection of babies, but less false positives
    - model_v3.pth: File length normalization to 7s results in better detection of babies, but more false positives
"""

# Define flags
RETRAIN = False  # make this true when you wish to generate a new model
USE_SPECIFIC_MODEL = True  # make this true when you wish to load a specific model (MODEL_TO_USE) - exclusive with RETRAIN
NORMALIZE_AUDIO_LENGTH = False  # make this true when you wish to normalize the length of audio files (AUDIO_FILE_LENGTH)
VERBOSE = False  # make this true when you wish to display additional information in the console when running the program

# Determine if the application is running as a PyInstaller bundle
if getattr(sys, 'frozen', False):
    USE_SPECIFIC_MODEL = True  # force the use of a specific model when running as a PyInstaller bundle
    RETRAIN = False  # force the program to not retrain when running as a PyInstaller bundle
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_PATH = os.path.join(application_path, 'models')  # path to the folder containing the models


# Entry point for the program
def main():
    baby_detector = ac.AudioClassifier(NUM_MFCC_COEFFS, MODELS_PATH)  # Initialize the neural network for audio classification

    # Create or load a model for the audio classifier
    if RETRAIN:  # This runs when a new model is to be generated

        # Prepare training data
        if NORMALIZE_AUDIO_LENGTH:  # normalize audio length if specified
            # Load baby crying sounds & label them 1
            crying_features, crying_labels = load_data(CRYING_PATH, 1,
                                                       SAMPLE_RATE, NUM_MFCC_COEFFS,
                                                       normalize_to_length=AUDIO_FILE_LENGTH)
            # Load other noise & label them 0
            noise_features, noise_labels = load_data(NOISE_PATH, 0,
                                                     SAMPLE_RATE, NUM_MFCC_COEFFS,
                                                     normalize_to_length=AUDIO_FILE_LENGTH)
        else:  # no audio length normalization
            crying_features, crying_labels = load_data(CRYING_PATH, 1, SAMPLE_RATE, NUM_MFCC_COEFFS)  # Load baby crying sounds & label them 1
            noise_features, noise_labels = load_data(NOISE_PATH, 0, SAMPLE_RATE, NUM_MFCC_COEFFS)  # Load other noise & label them 0

        baby_detector.new_model(crying_features, noise_features, crying_labels, noise_labels, verbose=VERBOSE)  # Train the model

    else:  # This runs when using a previously generated model
        baby_detector.load_model(MODEL_TO_USE, verbose=VERBOSE)  # Load the model

    # Get input and output devices (NOTE: check these indices using sd.query_devices())
        # TODO: Add a GUI to select the input and output devices before starting the filter
    # input_device = 0  # Input device index (microphone / computer audio)
    # output_device = 2  # Output device index (speakers / headphones)
    device_selector = gui.DeviceSelector()  # Initialize the GUI for selecting input and output devices
    input_device, output_device = device_selector.select_devices()  # Prompt user for input and output devices

    plotter = dp.AudioDataPlotter(NUM_MFCC_COEFFS, MFCC_RANGE)  # Initialize the audio data plotter

    # Initialize the audio processor
    ap = AudioProcessor(baby_detector, plotter, SAMPLE_RATE, BLOCK_SIZE, NUM_MFCC_COEFFS, N_FFT, HOP_LENGTH,
                        input_device, output_device, verbose=VERBOSE)

    ap.open_streams()  # Open the input audio stream and start processing audio
