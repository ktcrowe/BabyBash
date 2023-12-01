# Tools for audio processing

from scipy import signal  # For signal processing

import numpy as np  # For numerical operations
import librosa  # For audio feature extraction
import torch  # PyTorch library for neural networks
import sounddevice as sd  # For audio recording and processing


# Class for processing audio input data
class AudioProcessor:
    def __init__(self, model, data_plotter, sample_rate, block_size, n_mfcc, n_fft, hop_length,
                 input_device, output_device, verbose=False):
        self.model = model
        self.data_plotter = data_plotter
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_device = output_device
        self.input_device = input_device
        self.verbose = verbose

        self.detected_count = 0  # the number of times a baby has been detected (used for comparing models)
        self.output_stream = None  # the output stream for playing back audio
        self.detection_buffer = [0] * 50  # buffer for storing previous predictions (ex: 50 * 20ms = 1 second)
        self.baby_previously_detected = False  # Flag for detecting when a baby was detected in the buffer
        self.fade_in_length = 5  # Number of blocks for fade-in
        self.fade_out_length = 100  # Number of blocks for fade-out
        self.fade_in_counter = 0  # Counter for fade-in
        self.fade_out_counter = 0  # Counter for fade-out

        # Initialize the filter's threshold to reduce false positive filtering
        self.threshold_length = 50  # Number of blocks to check for a baby in the buffer
        self.threshold = 5  # Number of blocks with a baby in the threshold to trigger filtering

        self.lp_filter_b, self.lp_filter_a = initialize_low_pass_filter(cutoff_freq=1000, sample_rate=self.sample_rate)
        self.filter_state = signal.lfilter_zi(self.lp_filter_b, self.lp_filter_a)

    # Function called by the audio stream when new data is available
    def audio_callback(self, indata, outdata, frames, time, status):
        try:
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

            # Update GUI when a crying baby is detected
            prediction_text = "Baby crying detected!" if predicted.item() == 1 else ""
            self.data_plotter.update_prediction_text(prediction_text)

            # Process audio block with filters
            baby_detected = predicted.item() == 1
            self.update_detection_buffer(baby_detected)
            fade_factor = self.calculate_fade_factor()
            self.update_filter_activity_text(fade_factor)

            if fade_factor > 0:
                # Apply the filter with the maintained filter state
                filtered_audio, self.filter_state = signal.lfilter(self.lp_filter_b, self.lp_filter_a, indata[:, 0],
                                                                   zi=self.filter_state)
                outdata[:] = (fade_factor * filtered_audio + (1 - fade_factor) * indata[:, 0]).reshape(-1, 1)
            else:
                outdata[:] = indata

            # Display the number of times a baby has been detected if verbose mode is enabled
            if self.verbose:
                self.detected_count += 1 if predicted.item() == 1 else 0
                print(f'Detected {self.detected_count} times.')

        except Exception as e:
            print(f'Error during audio callback: {e}')

    # Update the buffer of previous blocks with the current detection
    def update_detection_buffer(self, current_detection):
        # Update detection buffer with the current detection
        self.detection_buffer.pop(0)  # Remove oldest detection
        self.detection_buffer.append(current_detection)  # Add the newest detection

    # Calculate the fade factor for the filter (this is where fade-in/out is implemented)
    def calculate_fade_factor(self):
        # Calculate recent detections within the threshold length
        recent_detections = sum(self.detection_buffer[-self.threshold_length:])
        if recent_detections > self.threshold:
            self.fade_in_counter = min(self.fade_in_counter + 1, self.fade_in_length)
            self.fade_out_counter = 0  # Interrupt fade-out if baby is detected again
        else:
            self.fade_in_counter = max(self.fade_in_counter - 1, 0)
            if self.fade_out_counter == 0 and self.baby_previously_detected:
                self.fade_out_counter = self.fade_out_length  # Start fade-out if no new detection

        # Calculate the filter's fade factor based on the fade counters
        if self.fade_in_counter > 0:
            return self.fade_in_counter / self.fade_in_length
        return self.fade_out_counter / self.fade_out_length

    # Update the text on the plot to indicate whether the filter is active
    def update_filter_activity_text(self, fade_factor):
        filter_active_text = "Baby bashing!" if fade_factor > 0 else ""
        self.data_plotter.update_filter_activity_text(filter_active_text, fade_factor)

    # Open the input and output audio streams and start processing audio
    def open_streams(self):
        try:
            with sd.Stream(callback=self.audio_callback, dtype='float32', channels=1,
                           samplerate=self.sample_rate, blocksize=self.block_size,
                           device=(self.input_device, self.output_device)) as stream:
                self.data_plotter.animate()  # Start the animation of the plot
        except Exception as e:
            print(f'Error opening streams: {e}')


# Compute Mel-frequency cepstral coefficients (MFCCs) from an audio buffer
def compute_mfcc(indata, sample_rate, n_mfcc, n_fft, hop_length):
    # Compute the MFCCs for the input data, transpose it, and take the mean across time
    mfccs = librosa.feature.mfcc(y=indata[:, 0], sr=sample_rate,
                                 n_mfcc=n_mfcc, n_fft=n_fft,
                                 hop_length=hop_length).T
    return np.mean(mfccs, axis=0)


# Create a low-pass filter that crying babies will be run through
def initialize_low_pass_filter(cutoff_freq, sample_rate, order=5):
    """
    Initialize a low-pass filter.

    Parameters:
    - cutoff_freq: Cutoff frequency of the low-pass filter (in Hz).
    - fs: Sampling rate (in Hz).
    - order: Order of the filter.

    Returns:
    - b, a: Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    filter_coefs = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return filter_coefs[0], filter_coefs[1]


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
