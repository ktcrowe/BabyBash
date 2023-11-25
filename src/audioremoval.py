import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks


class AudioRemover:
    def __init__(self, dataset_path, sample_rate, batch_size=10, num_top_freqs=20, verbose=False):
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_top_freqs = num_top_freqs
        self.verbose = verbose
        self.key_frequencies_aggregate = []
        self.key_frequencies = None
        self.key_freqs_file = 'data/key_freqs'

    def process_batches(self):
        """
        Process the dataset in batches to compute average key frequencies.
        """
        audio_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.wav')]
        total_files = len(audio_files)
        num_batches = total_files // self.batch_size + (1 if total_files % self.batch_size else 0)
        print(f'AudioRemover: Processing {total_files} audio files in {num_batches} batches...') if self.verbose else None

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_files)
            self._process_batch(audio_files[start_idx:end_idx])
            print(f'AudioRemover: Processed batch {i + 1} of {num_batches}') if self.verbose else None

        # Select the most significant key frequencies
        self.key_frequencies = self._select_significant_frequencies()
        self.save_key_frequencies()

        print(f'AudioRemover: Processed {total_files} audio files in {num_batches} batches.') if self.verbose else None

    def _process_batch(self, batch_files):
        """
        Process a single batch of audio files.
        """
        audio_data = []
        for file in batch_files:
            file_path = os.path.join(self.dataset_path, file)
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            audio_data.append(audio)

        concatenate_audio = np.concatenate(audio_data)

        # Compute the magnitude spectrum
        magnitude_spectrum = np.abs(np.fft.rfft(concatenate_audio))

        # Find peaks in the magnitude spectrum
        peaks, _ = find_peaks(magnitude_spectrum)

        # Identify the key frequencies at the peaks
        freqs = np.fft.rfftfreq(len(concatenate_audio), d=1./self.sample_rate)
        key_frequencies = freqs[peaks]
        key_frequencies = key_frequencies[key_frequencies <= 20000]  # Filter out frequencies above 20 kHz
        self.key_frequencies_aggregate.append(key_frequencies)

    def _select_significant_frequencies(self):
        """
        Select the most significant frequencies based on their occurrence.
        """
        # Flatten the list of key frequencies
        all_frequencies = np.concatenate(self.key_frequencies_aggregate)

        # Count the occurrence of each frequency
        freqs, counts = np.unique(all_frequencies, return_counts=True)

        # Select the top frequencies based on occurrence
        top_freq_indices = np.argsort(counts)[::-1][:self.num_top_freqs]
        top_freqs = freqs[top_freq_indices]

        return top_freqs

    # save the key frequencies to a file
    def save_key_frequencies(self):
        """
        Save the key frequencies to a file.
        """
        np.savetxt(self.key_freqs_file, self.key_frequencies)
        print(f"Key frequencies saved to {self.key_freqs_file}")

    # load the key frequencies from a file
    def load_key_frequencies(self):
        """
        Load key frequencies from a file.
        """
        try:
            self.key_frequencies = np.loadtxt(self.key_freqs_file)
            print(f"Key frequencies loaded from {self.key_freqs_file}")
        except IOError:
            print(f"Could not load key frequencies from {self.key_freqs_file}")

    def plot_key_frequencies(self):
        """
        Plot the key frequencies.
        """
        plt.figure()
        plt.plot(self.key_frequencies, 'o')
        plt.xlabel('Frequency Index')
        plt.ylabel('Frequency (Hz)')
        plt.title('Key Frequencies in Dataset')
        plt.show()
