# Import required libraries
import audioclassification as ac  # Neural network for audio classification
import dataplotting as dp  # Tools for plotting and visualization of audio data
import audioremoval as ar  # Tools for removing audio from the input stream
from filemanagement import load_files  # Tools for file management
from audioprocessing import AudioProcessor  # Tools for audio processing

import sounddevice as sd  # For audio recording and processing

# Define constant values for audio processing
SAMPLE_RATE = 44100  # Sample rate in Hz for audio processing
BLOCK_SIZE = 882  # Block size for the audio stream buffer (882 creates 20ms blocks)
NUM_MFCC_COEFFS = 26  # Number of Mel-frequency cepstral coefficients (MFCCs) to compute
MFCC_RANGE = 10  # Range of values for the y-axis on the plot (initially was 275, but reduced to reflect normalized MFCCs)
N_FFT = BLOCK_SIZE  # Fast Fourier Transform (FFT) window size set to block size
HOP_LENGTH = N_FFT // 4  # Hop length for the FFT, typically 1/4th of the FFT window size
AUDIO_FILE_LENGTH = 7  # Length of audio files in seconds (used for normalizing audio length)
QUALITY_FACTOR = 30  # Quality factor for notch filters

# Define paths
CRYING_PATH = 'data/crying'  # audio files of crying babies
NOISE_PATH = 'data/noise'  # audio files of various other noises (not babies crying)
MODEL_TO_USE = 'model_v1.pth'  # the model to use for detecting crying babies (if USE_SPECIFIC_MODEL is True)

# Define flags
RETRAIN = False  # make this true when you wish to generate a new model
CALCULATE_KEY_FREQUENCIES = False  # make this true when you wish to calculate the average key frequencies from the dataset
USE_SPECIFIC_MODEL = True  # make this true when you wish to load a specific model (MODEL_TO_USE) - exclusive with RETRAIN
NORMALIZE_AUDIO_LENGTH = False  # make this true when you wish to normalize the length of audio files (AUDIO_FILE_LENGTH)
VERBOSE = True  # make this true when you wish to display additional information in the console when running the program

# main script
if __name__ == '__main__':
    baby_detector = ac.AudioClassifier(NUM_MFCC_COEFFS)  # Initialize the neural network for audio classification
    baby_remover = ar.AudioRemover(CRYING_PATH, SAMPLE_RATE, verbose=VERBOSE)  # Initialize the audio remover
    print(sd.query_devices()) if VERBOSE else None  # DEBUG

    if CALCULATE_KEY_FREQUENCIES:  # This runs when the average key frequencies are to be calculated
        baby_remover.process_batches()  # Compute the average key frequencies from the dataset
        baby_remover.plot_key_frequencies()  # DEBUG
    key_frequencies = baby_remover.load_key_frequencies()  # Load the average key frequencies of the dataset

    if RETRAIN:  # This runs when a new model is to be generated

        # Prepare training data
        if NORMALIZE_AUDIO_LENGTH:  # normalize audio length if specified
            # Load baby crying sounds & label them 1
            crying_features, crying_labels = load_files(CRYING_PATH, 1,
                                                        SAMPLE_RATE, NUM_MFCC_COEFFS,
                                                        normalize_to_length=AUDIO_FILE_LENGTH)
            # Load other noise & label them 0
            noise_features, noise_labels = load_files(NOISE_PATH, 0,
                                                      SAMPLE_RATE, NUM_MFCC_COEFFS,
                                                      normalize_to_length=AUDIO_FILE_LENGTH)
        else:  # no audio length normalization
            crying_features, crying_labels = load_files(CRYING_PATH, 1, SAMPLE_RATE, NUM_MFCC_COEFFS)  # Load baby crying sounds & label them 1
            noise_features, noise_labels = load_files(NOISE_PATH, 0, SAMPLE_RATE, NUM_MFCC_COEFFS)  # Load other noise & label them 0

        baby_detector.new_model(crying_features, noise_features, crying_labels, noise_labels, verbose=VERBOSE)  # Train the model

    else:  # This runs when using a previously generated model
        baby_detector.load_model(MODEL_TO_USE, verbose=VERBOSE)  # Load the model

    plotter = dp.AudioDataPlotter(NUM_MFCC_COEFFS, MFCC_RANGE)  # Initialize the audio data plotter
    # Initialize the audio processor
    ap = AudioProcessor(baby_detector, plotter, SAMPLE_RATE, BLOCK_SIZE, NUM_MFCC_COEFFS, N_FFT, HOP_LENGTH,
                        key_frequencies, QUALITY_FACTOR, 0, 2, verbose=VERBOSE)
    ap.open_streams()  # Open the input audio stream and start processing audio
