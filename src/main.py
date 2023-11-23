# Import required libraries
import sounddevice as sd  # For audio recording and processing
import numpy as np  # For numerical operations
import librosa  # For audio feature extraction
import torch  # PyTorch library for neural networks
import matplotlib.pyplot as plt  # For plotting data
from matplotlib.animation import FuncAnimation  # For animating the plot

import audioclassification as ac  # Neural network for audio classification
import dataplot as dp  # Tools for plotting and visualization of audio data
from filemanagement import load_files  # Tools for file management
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

# Define flags
RETRAIN = False  # make this true when you wish to generate a new model
USE_SPECIFIC_MODEL = True  # make this true when you wish to load a specific model (MODEL_TO_USE) - exclusive with RETRAIN
NORMALIZE_AUDIO_LENGTH = False  # make this true when you wish to normalize the length of audio files (AUDIO_FILE_LENGTH)
VERBOSE = True  # make this true when you wish to display additional information in the console when running the program

# prediction_text = ''  # the text to be displayed on the GUI to indicate whether a baby is detected
# detected_count = 0  # the number of times a baby has been detected (used for comparing models)


# # Compute Mel-frequency cepstral coefficients (MFCCs) from an audio buffer
# def compute_mfcc(indata):
#     # Compute the MFCCs for the input data, transpose it, and take the mean across time
#     mfccs = librosa.feature.mfcc(y=indata[:, 0], sr=SAMPLE_RATE,
#                                  n_mfcc=NUM_MFCC_COEFFS, n_fft=N_FFT,
#                                  hop_length=HOP_LENGTH).T
#     return np.mean(mfccs, axis=0)


# # Update the plot with new data
# def update_plot(frame):
#     """
#     This function gets called by the animation framework with a new frame
#     (essentially an increment in time/intervals).
#     """
#     line.set_ydata(mfccs)  # Set new y-data for the line object
#     text_element.set_text(prediction_text)  # Update the text based on the prediction
#
#     """
#     Return the line object that has been modified, which is necessary for blitting to work.
#     "Blitting" means that the animation only re-draws the parts that have actually changed to improve the animation
#     """
#     return line, text_element


# # Function called by the audio stream when new data is available
# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     global mfccs, baby_detector, scaler, prediction_text, detected_count
#
#     mfccs = compute_mfcc(indata)  # Compute MFCCs from the input data
#     mfccs = scaler.transform([mfccs])  # Standardize the MFCCs (Note: scaler expects a 2D array)
#     mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)  # Convert to tensor
#
#     # Make a prediction (is a baby crying or not?)
#     baby_detector.eval()  # Set the model to evaluation mode
#     with torch.no_grad():
#         """
#         The above line tells PyTorch not to compute gradients in order to improve performance, since this does not
#         need to be done when we are not training the model.
#         """
#         outputs = baby_detector(mfccs_tensor)  # run the current MFCCs of the input data through the network
#         _, predicted = torch.max(outputs.data, 1)  # Retrieve the index of the larger value ('_' ignores the actual value)
#
#     # Display on the GUI when a crying baby is detected
#     prediction_text = "Baby crying detected!" if predicted.item() == 1 else ""
#     detected_count += 1 if predicted.item() == 1 else 0
#     print(f'Detected {detected_count} times.')  # DEBUG


# main script
if __name__ == '__main__':
    # # Define the plot to display the input audio in real-time
    # plt.style.use('fast')  # Set a fast plotting style for real-time updates
    # fig, ax = plt.subplots()  # Initialize the plot figure and axis
    # x_axis = np.arange(NUM_MFCC_COEFFS)  # Generate x-axis values corresponding to the MFCC coefficients
    # line, = ax.plot(x_axis, np.zeros(NUM_MFCC_COEFFS))  # Create an initial line object with x-axis and y-axis data for the plot
    #
    # # Set the y-axis and x-axis limits for the plot
    # ax.set_ylim(-MFCC_RANGE, MFCC_RANGE)
    # ax.set_xlim(0, NUM_MFCC_COEFFS-1)
    #
    # # Label the x-axis and y-axis of the plot
    # ax.set_xlabel('MFCC Coefficients')
    # ax.set_ylabel('Amplitude')
    #
    # ax.set_title('Real-time MFCC')  # Set the title for the plot
    #
    # # Initialize the text element on the plot to display when a baby is crying
    # text_element = ax.text(0.5, 0.1, '', horizontalalignment='center', verticalalignment='center',
    #                        transform=ax.transAxes)
    #
    # mfccs = np.zeros(NUM_MFCC_COEFFS)  # Initialize an array to hold the MFCC data, initially filled with zeros

    baby_detector = ac.AudioClassifier(NUM_MFCC_COEFFS)  # Initialize the neural network for audio classification
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

    scaler = baby_detector.scaler  # Load the scaler used to normalize the MFCCs

    plotter = dp.AudioDataPlotter(NUM_MFCC_COEFFS, MFCC_RANGE)  # Initialize the audio data plotter
    # Initialize the audio processor
    ap = AudioProcessor(baby_detector, plotter, SAMPLE_RATE, BLOCK_SIZE, NUM_MFCC_COEFFS, N_FFT, HOP_LENGTH, verbose=VERBOSE)
    ap.open_input_stream()  # Open the input audio stream and start processing audio
    # with sd.InputStream(callback=ap.audio_callback, dtype='float32', channels=1,
    #                     samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
    #     # Create an animation object that will update the plot in real-time
    #     ani = FuncAnimation(fig, update_plot, blit=True, interval=10, cache_frame_data=False)
    #     plt.show()  # Display the matplotlib plot and start the animation
