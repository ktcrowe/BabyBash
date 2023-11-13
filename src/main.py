# Import required libraries
import os
import sounddevice as sd  # For audio recording and processing
import numpy as np  # For numerical operations
import librosa  # For audio feature extraction
import torch  # PyTorch library for neural networks
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch.optim as optim  # Optimization functions
import matplotlib.pyplot as plt  # For plotting data
from matplotlib.animation import FuncAnimation  # For creation animations on the plot
from torch.utils.data import DataLoader, TensorDataset  # For handling datasets
from sklearn.model_selection import train_test_split  # For splitting datasets
from sklearn.preprocessing import StandardScaler  # For feature scaling (normalization)
from joblib import dump, load  # For saving and loading objects (the scaler in this case)

# Define constant values for audio processing
SAMPLE_RATE = 44100  # Sample rate in Hz for audio processing
BLOCK_SIZE = 882  # Block size for the audio stream buffer (882 creates 20ms blocks)
NUM_MFCC_COEFFS = 26  # Number of Mel-frequency cepstral coefficients (MFCCs) to compute
MFCC_RANGE = 10  # Range of values for the y-axis on the plot (initially was 275, but changed to 2 to reflect standardized MFCCs)
N_FFT = BLOCK_SIZE  # Fast Fourier Transform (FFT) window size set to block size
HOP_LENGTH = N_FFT // 4  # Hop length for the FFT, typically 1/4th of the FFT window size

# Define paths
CRYING_PATH = 'data/crying'  # audio files of crying babies
NOISE_PATH = 'data/noise'  # audio files of various other noise (not babies crying)
MODELS_PATH = 'models'  # folder where AI models are saved

RETRAIN = False  # make this true when you wish to generate a new model
prediction_text = ''  # the text to be displayed on the GUI to indicate whether a baby is detected


# Define a simple neural network
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        """
        The two parameters taken by each layer are the number of input features and the number of output features.
        The first layer takes the MFCCs, and eventually this is reduced to two nodes on the output layer
        to signify the probability that a baby is crying in the input audio.
        """
        self.fc1 = nn.Linear(NUM_MFCC_COEFFS, 128)  # linear layer 1 - input layer
        self.fc2 = nn.Linear(128, 64)  # linear layer 2 - hidden layer
        self.fc3 = nn.Linear(64, 2)  # linear layer 3 - output layer

    # Forward pass through the network
    def forward(self, x):
        # This applies the Rectified Linear Unit function (ReLU)
        #   f(x) = max(0, x)
        #   In other words, any negative values from the output of a given layer are changed to 0.
        x = F.relu(self.fc1(x))  # apply ReLU to layer 1 output
        x = F.relu(self.fc2(x))  # apply ReLU to layer 2 output
        x = self.fc3(x)  # output layer
        return x


# Load and label the files for use in training the network
def load_files(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            path = os.path.join(folder, filename)
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)  # Load the audio file
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=NUM_MFCC_COEFFS)  # Extract MFCCs
            mfcc = mfcc.mean(axis=1)  # Take the mean of the MFCCs over time
            features.append(mfcc)
            labels.append(label)
    return features, labels


# Define a function to compute Mel-frequency cepstral coefficients (MFCCs) from an audio buffer
def compute_mfcc(indata):
    # Compute the MFCCs for the input data, transpose it, and take the mean across time
    mfccs = librosa.feature.mfcc(y=indata[:, 0], sr=SAMPLE_RATE,
                                 n_mfcc=NUM_MFCC_COEFFS, n_fft=N_FFT,
                                 hop_length=HOP_LENGTH).T
    return np.mean(mfccs, axis=0)


# Define a function to update the plot with new data
def update_plot(frame):
    """
    This function gets called by the animation framework with a new frame
    (essentially an increment in time/intervals).
    """
    line.set_ydata(mfccs)  # Set new y-data for the line object
    text_element.set_text(prediction_text)  # Update the text based on the prediction

    """
    Return the line object that has been modified, which is necessary for blitting to work.
    "Blitting" means that the animation only re-draws the parts that have actually changed to improve the animation
    """
    return line, text_element


# Define a callback function that will be called by the audio stream when new data is available
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    global mfccs, model, scaler, prediction_text

    mfccs = compute_mfcc(indata)  # Compute MFCCs from the input data
    mfccs = scaler.transform([mfccs])  # Standardize the MFCCs (Note: scaler expects a 2D array)
    mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32)  # Convert to tensor

    # Make a prediction (is a baby crying or not?)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        """
        The above line tells PyTorch not to compute gradients in order to improve performance, since this does not
        need to be done when we are not training the model.
        """
        outputs = model(mfccs_tensor)  # run the current MFCCs of the input data through the network
        _, predicted = torch.max(outputs.data, 1)  # Retrieve the index of the larger value ('_' ignores the actual value)

    # Display on the GUI when a crying baby is detected
    prediction_text = "Baby crying detected!" if predicted.item() == 1 else ""


# Count the amount of files in a given directory (used for automatically naming models when loading/saving)
def count_files_in_folder(folder_path):
    if not os.path.exists(folder_path):
        return "Folder not found."

    file_count = 0
    for entry in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, entry)):
            file_count += 1

    return file_count


# main script
if __name__ == '__main__':
    # Define the plot to display the input audio in real-time
    plt.style.use('fast')  # Set a fast plotting style for real-time updates
    fig, ax = plt.subplots()  # Initialize the plot figure and axis
    x_axis = np.arange(NUM_MFCC_COEFFS)  # Generate x-axis values corresponding to the MFCC coefficients
    line, = ax.plot(x_axis,
                    np.zeros(NUM_MFCC_COEFFS))  # Create an initial line object with x-axis and y-axis data for the plot

    # Set the y-axis and x-axis limits for the plot
    ax.set_ylim(-MFCC_RANGE, MFCC_RANGE)
    ax.set_xlim(0, NUM_MFCC_COEFFS - 1)

    # Label the x-axis and y-axis of the plot
    ax.set_xlabel('MFCC Coefficients')
    ax.set_ylabel('Amplitude')

    # Set the title for the plot
    ax.set_title('Real-time MFCC')

    # Initialize a text element on the plot to display when a baby is crying
    text_element = ax.text(0.5, 0.1, '', horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes)

    # Initialize an array to hold the MFCC data, initially filled with zeros
    mfccs = np.zeros(NUM_MFCC_COEFFS)

    # Initialize the neural network
    model = AudioClassifier()

    if not RETRAIN:  # This runs when using a previously generated model
        model.load_state_dict(torch.load(f'{MODELS_PATH}/model_v{count_files_in_folder(MODELS_PATH)-1}.pth'))  # load an existing model
        scaler = load('models/scaler.joblib')  # load an existing scaler

    else:  # This runs when a new model is to be generated
        # Prepare training data
        crying_features, crying_labels = load_files(CRYING_PATH, 1)  # Load baby crying sounds & label them 1
        noise_features, noise_labels = load_files(NOISE_PATH, 0)  # Load other noise & label them 0

        # create a 1:1 ration of clean to dirty audio files
        noise_features, noise_labels = noise_features[:len(crying_features)], noise_labels[:len(crying_labels)]
        # print(len(crying_features), len(noise_features))  # DEBUG

        # Combine and shuffle the data
        features = np.array(crying_features + noise_features)
        labels = np.array(crying_labels + noise_labels)
        # print(len(features), len(labels))  # DEBUG

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        dump(scaler, 'models/scaler.joblib')  # save the scaler

        """
        Split the dataset
        test_size=0.2 creates an 80/20 split between training and testing data
        random_state=42 is a seed that ensures this split is the same every time
        (42 is an arbitrary number that is popular amongst nerds)
        """
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Convert NumPy arrays to PyTorch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 10  # the network will be run through 10 times
        for epoch in range(num_epochs):
            for data, labels in train_loader:
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)

                # Backward and optimize AKA backpropagation
                optimizer.zero_grad()  # clear gradients from the previous iteration (first step of backpropagation)
                loss.backward()  # compute the gradients for this iteration
                optimizer.step()  # update the weights and biases of the network using the calculated gradients

                """
                What is a gradient, you may ask? It is simply the derivative of the loss function.
                Since the loss function measures how well the model is performing, the gradient is the rate of change
                in the loss function - or the change in effectiveness of the predictive model.
                """

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # Evaluate the model
        model.eval()
        with torch.no_grad():  # gradient calculations not needed on the trained network
            correct = 0  # the count of correct guesses made by the network
            total = 0  # the count of total guesses made by the network
            for data, labels in test_loader:
                outputs = model(data)  # send the current data through the network
                _, predicted = torch.max(outputs.data, 1)  # get the prediction
                total += labels.size(0)  # increment guess count
                correct += (predicted == labels).sum().item()  # increment correct guess count

        print(f'Accuracy of the model on the test data: {100 * correct / total}%')  # print accuracy after all epochs

        torch.save(model.state_dict(), f'{MODELS_PATH}/model_v{count_files_in_folder(MODELS_PATH)}.pth')  # Save the model

    # Use a context manager to open an audio input stream with the specified parameters
    with sd.InputStream(callback=audio_callback, dtype='float32', channels=1,
                        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
        # Create an animation object that will update the plot in real-time
        ani = FuncAnimation(fig, update_plot, blit=True, interval=10, cache_frame_data=False)
        plt.show()  # Display the matplotlib plot and start the animation
