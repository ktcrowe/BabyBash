# Tools for audio processing

import numpy as np  # For numerical operations
import librosa  # For audio feature extraction
import torch  # PyTorch library for neural networks
import sounddevice as sd  # For audio recording and processing


class AudioProcessor:
    def __init__(self, model, data_plotter, sample_rate, block_size, n_mfcc, n_fft, hop_length, verbose=False):
        self.model = model
        self.data_plotter = data_plotter
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.verbose = verbose
        self.detected_count = 0  # the number of times a baby has been detected (used for comparing models)

    # Function called by the audio stream when new data is available
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)

        # Compute MFCCs from the input data
        mfccs = compute_mfcc(indata, sample_rate=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                             hop_length=self.hop_length)
        mfccs = self.model.scaler.transform([mfccs])  # Standardize the MFCCs (Note: scaler expects a 2D array)
        self.data_plotter.update_mfcc_data(mfccs)  # Update the plot with the new MFCCs
        mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)  # Convert to tensor

        # Make a prediction (is a baby crying or not?)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            """
            The above line tells PyTorch not to compute gradients in order to improve performance, since this does not
            need to be done when we are not training the model.
            """
            outputs = self.model(mfccs_tensor)  # run the current MFCCs of the input data through the network
            _, predicted = torch.max(outputs.data,
                                     1)  # Retrieve the index of the larger value ('_' ignores the actual value)

        # # Display on the GUI when a crying baby is detected
        prediction_text = "Baby crying detected!" if predicted.item() == 1 else ""
        self.data_plotter.update_prediction_text(prediction_text)

        # Display the number of times a baby has been detected if verbose mode is enabled
        if self.verbose:
            self.detected_count += 1 if predicted.item() == 1 else 0
            print(f'Detected {self.detected_count} times.')

    def open_input_stream(self):
        with sd.InputStream(callback=self.audio_callback, dtype='float32', channels=1,
                            samplerate=self.sample_rate, blocksize=self.block_size):
            print(f'Listening for audio at {self.sample_rate} Hz')  # DEBUG
            self.data_plotter.animate()  # Start the animation of the plot


# Compute Mel-frequency cepstral coefficients (MFCCs) from an audio buffer
def compute_mfcc(indata, sample_rate, n_mfcc, n_fft, hop_length):
    # Compute the MFCCs for the input data, transpose it, and take the mean across time
    mfccs = librosa.feature.mfcc(y=indata[:, 0], sr=sample_rate,
                                 n_mfcc=n_mfcc, n_fft=n_fft,
                                 hop_length=hop_length).T
    return np.mean(mfccs, axis=0)


# normalize audio files to a target length in seconds
def normalize_audio_length(audio_path, target_length, sample_rate):
    """
    Normalize the length of an audio file to a target length in seconds.

    Parameters:
    audio_path (str): Path to the audio file.
    target_length (int): Desired length of the audio in seconds.
    sample_rate (int): Sample rate for the audio file.

    Returns:
    numpy.ndarray: Normalized audio data.
    """

    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    # Calculate the target number of samples
    target_samples = target_length * sample_rate

    # Get the number of samples in the audio file
    current_samples = audio.shape[0]

    # Pad or truncate the audio data
    if current_samples < target_samples:
        # Pad with zeros (silence) if the audio is shorter than the target length
        pad_length = target_samples - current_samples
        padded_audio = np.pad(audio, (0, pad_length), mode='constant')
    else:
        # Truncate the audio if it's longer than the target length
        padded_audio = audio[:target_samples]

    return padded_audio
