# The model for the audio classification task is defined here.

import numpy as np  # For numerical operations
import torch  # PyTorch library for neural networks
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Neural network functions
import torch.optim as optim  # Optimization functions
from torch.utils.data import DataLoader, TensorDataset  # For handling datasets
from sklearn.model_selection import train_test_split  # For splitting datasets
from sklearn.preprocessing import StandardScaler  # For feature scaling (MFCC normalization)
from joblib import dump, load  # For saving and loading objects (the scaler in this case)

from filemanagement import count_files_in_folder  # Tools for file management

MODELS_PATH = 'models'  # folder where AI models are saved


# A simple neural network to detect whether a baby is crying in an audio stream
class AudioClassifier(nn.Module):
    def __init__(self, n_mfcc):
        super(AudioClassifier, self).__init__()
        self.scaler = None
        """
        The two parameters taken by each layer are the number of input features and the number of output features.
        The first layer takes the MFCCs, and eventually this is reduced to two nodes on the output layer
        to signify the probability that a baby is crying in the input audio.
        """
        self.fc1 = nn.Linear(n_mfcc, 128)  # linear layer 1 - input layer
        self.fc2 = nn.Linear(128, 64)  # linear layer 2 - hidden layer
        self.fc3 = nn.Linear(64, 2)  # linear layer 3 - output layer

    # Forward pass through the network
    def forward(self, x):
        """
        This applies the Rectified Linear Unit function (ReLU)
            f(x) = max(0, x)
        In other words, any negative values from the output of a given layer are changed to 0.
        """
        x = F.relu(self.fc1(x))  # apply ReLU to layer 1 output
        x = F.relu(self.fc2(x))  # apply ReLU to layer 2 output
        x = self.fc3(x)  # output layer
        return x

    # Train a new model
    def new_model(self, positive_features, negative_features, positive_labels, negative_labels,
                  test_size=0.2, random_state=42, num_epochs=10, verbose=False):
        print(f'Training new model...') if verbose else None

        # create a 1:1 ratio of positive to negative audio files
        #   TODO: Try shuffling this list before the slice to get different results.
        if len(positive_features) < len(negative_features):  # more negative samples than positive samples
            negative_features, negative_labels = negative_features[:len(positive_features)], negative_labels[:len(positive_labels)]
        elif len(negative_features) < len(positive_features):  # more positive samples than negative samples
            positive_features, positive_labels = positive_features[:len(negative_features)], positive_labels[:len(negative_labels)]
        # print(len(crying_features), len(noise_features))  # DEBUG

        # Combine and shuffle the data
        features = np.array(positive_features + negative_features)
        labels = np.array(positive_labels + negative_labels)
        # print(len(features), len(labels))  # DEBUG

        # Normalize features (MFCCs) with a scaler
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        dump(self.scaler, 'models/scaler.joblib')  # save the scaler
        # TODO: Allow for multiple scalers to be saved (one for each model)

        """
        Split the dataset
            - test_size=0.2 creates an 80/20 split between training and testing data
            - random_state=42 is a seed that ensures this split is the same every time
                (42 is an arbitrary number that is popular amongst nerds)
        """
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size,
                                                            random_state=random_state)

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
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Training loop
        for epoch in range(num_epochs):  # run through the entire data set several times (AKA epochs)
            for data, labels in train_loader:
                # Forward pass
                outputs = self(data)
                loss = criterion(outputs, labels)

                # Backward and optimize AKA backpropagation
                optimizer.zero_grad()  # clear gradients from the previous epoch (first step of backpropagation)
                loss.backward()  # compute the gradients for this epoch
                optimizer.step()  # update the weights and biases of the network using the calculated gradients

                """
                What is a gradient, you may ask? It is the derivative of the loss function.
                Since the loss function measures how well the model is performing, the gradient is the rate of change
                in the loss function - or the change in effectiveness of the predictive model.
                This is essentially how the computer knows that the changes its making are improving the model.
                """

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

        # Evaluate the model
        self.eval()
        with torch.no_grad():  # gradient calculations not needed on the trained network
            correct = 0  # the count of correct guesses made by the network
            total = 0  # the count of total guesses made by the network
            for data, labels in test_loader:
                outputs = self(data)  # send the current test file through the network
                _, predicted = torch.max(outputs.data, 1)  # get the prediction index (ignore the actual value)
                total += labels.size(0)  # increment guess count
                correct += (predicted == labels).sum().item()  # increment correct guess count
                # TODO: The above two lines could be simplified, but they work, so I am afraid to touch them.

        print(f'Accuracy of the model on the test data: {100 * correct / total}%')  # print accuracy after all epochs
        torch.save(self.state_dict(),
                   f'{MODELS_PATH}/model_v{count_files_in_folder(MODELS_PATH)}.pth')  # Save the model

    # Load an existing model (defaults to the latest model)
    def load_model(self, model_name=None, verbose=False):
        if model_name:  # This runs when using a specific model
            self.load_state_dict(torch.load(f'{MODELS_PATH}/{model_name}'))  # TODO: Handle invalid model name
            print(f'Loaded model: {model_name}') if verbose else None
        else:  # This runs when using the latest model
            self.load_state_dict(torch.load(
                f'{MODELS_PATH}/model_v{count_files_in_folder(MODELS_PATH) - 1}.pth'))  # load latest model
            print(f'Loaded latest model: model_v{count_files_in_folder(MODELS_PATH) - 1}.pth') if verbose else None
        self.scaler = load('models/scaler.joblib')  # load an existing scaler
