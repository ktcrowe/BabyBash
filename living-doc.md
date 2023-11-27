# BabyBash


## Software Architecture

### Overview
**BabyBash** is designed for real-time audio processing to detect and filter out baby crying sounds. The architecture comprises the following major components:

#### 1. Audio Classification
- **Functionality:** Trains a new model or loads a pre-trained model to classify audio data as either baby crying or not baby crying. Achieved using a simple linear neural network architecture.
- **Interface:** Takes an input of MFCC features in a given block of audio data and outputs a classification.

#### 2. Audio Processing
- **Functionality:** Opens input and output streams, reads audio data from input and applies a filter when a baby is detected, then writes the filtered audio data to the output stream.
- **Interface:** Input and output of raw audio stream through specified devices.

#### 3. Data Plotting
- **Functionality:** Plots the input audio MFCC data in real-time to a graph. Indicate when a baby is detected and when the filter is being applied.
- **Interface:** Input of MFCC data, output of graph.

#### 4. File Management
- **Functionality:** Provide tools for loading data from files and managing data files.
- **Interface:** Input of file directories, output of data.

#### 5. GUI
- **Functionality:** Provides a user interface to allow user to choose audio devices and toggle the filter on and off.
- **Interface:** Interacts with the audio processing component to toggle the filter. Passes chosen devices to the audio processing component. Interacts with the data plotting component to display the toggle status.

### Assumptions
- The user has available audio devices to use for input and output.
- Audio input is continuous and stable.
- Sufficient computational resources for real-time processing.

### Alternative Architectures
#### 1. Cloud-based Processing
- **Pros:** Scalability, offloading processing load.
- **Cons:** Latency, privacy concerns, requires internet.

#### 2. Dedicated Hardware Processing
- **Pros:** Optimized performance, reliability.
- **Cons:** Higher cost, less flexibility.


## Software Design

### Overview

#### Audio Classification Module
- **Classes:** `AudioClassifier`
- **Responsibilities:** 
  - `AudioClassifier`: Load existing models, train new models if needed, classify input audio data in real-time.

#### Audio Processing Module
- **Classes:** `AudioProcessor`
- **Functions:** `compute_mfccs`, `initialize_filter`
- **Responsibilities:** 
  - `AudioProcessor`: Capture input audio, apply filter if needed, output filtered audio. Tells data plotter to update the animation accordingly.
  - `compute_mfccs`: Compute the MFCC features of a given audio signal.
  - `initialize_filter`: Initialize a filter to be applied to the audio signal.

#### Data Plotting Module
- **Classes:** `DataPlotter`
- **Responsibilities:** 
  - `DataPlotter`: Plot input MFCC data, animate in real-time, update relative information when needed (e.g. "Baby crying detected!").

#### File Management Module
- **Functions:** `load_data`, `count_files`
- **Responsibilities:** 
  - `load_data`: Load MFCC data from files implementing the audio processing module to do so.
  - `count_files`: Count the number of files in a given directory for naming new models appropriately.

#### GUI Module
- **Classes:** `DeviceSelector`
- **Functions:** `select_devices`
- **Responsibilities:** 
  - `DeviceSelector`: Builds the GUI to allow users to pick audio devices on program startup.
  - `handle_toggle_filter`: Initialize the `DeviceSelector` and return the selected devices.


## Coding Guidelines

### Overview
- **Language:** Python
- **Style Guide:** Flake8 (https://pypi.org/project/flake8/)
- **Explanation:** *(TODO: Add explanation for this choice.)*


## Process Description

### Risk Assessment
- **Risk (Cait):** Separation of baby crying from other sounds is too complex to be achieved in real-time.
  - **Likelihood:** High
  - **Impact:** High
  - **Evidence:** Since Deezer's "spleeter" library is not available to us, we would have to implement our own second NN using something akin to U-Net architecture. This would be a very complex task that is likely to not be feasible in the time we have.
  - **Reduction:** Experimented with different approaches to the problem of audio separation.
  - **Detection:** Audio can not be separated in real-time, or at all.
  - **Mitigation:** Apply a simple filter to the audio signal to muffle the sound of the baby crying.

- **Risk (Cait):** Detection of baby crying is not accurate enough to provide full coverage without false positives.
  - **Likelihood:** High
  - **Impact:** High
  - **Evidence:** The model is trained on a small dataset of only 1000 samples. This is not enough to provide full coverage of all possible baby crying sounds. The real-time nature of the application means that detections will be less accurate.
  - **Reduction:** Experimented with different approaches to the problem of audio classification, such as CNN architecture. This was not feasible given time constraints and lack of time axis in real-time audio data. Experimented with audio file length normalization and different amounts of MFCC coefficients to no avail.
  - **Detection:** The model is not able to detect baby crying in some cases. In other cases, the model detects baby crying when there is none.
  - **Mitigation:** Make use of fades and audio buffers to increase the window of audio filtering (filling in the blanks so to speak). Implement a threshold in the buffer to reduce false-positive filtering. Include more training data in the model.

- **(TODO: Israel's Risk)**
- **(TODO: Gwen's Risk)**
- **(TODO: Richard's Risk)**

### Project Schedule
- **Milestones:** Prototype completion, Alpha release, Beta release, Final release
  - **Prototype completion (REACHED 11/24/23):** Basic functionality of the application is implemented. The audio processing module is able to capture input audio and output filtered audio. The audio classification module is able to classify audio data as either baby crying or not baby crying. The data plotting module is able to plot input audio data in real-time. The file management module is able to load data from files. The GUI module is able to allow the user to choose input and output devices and toggle the filter. Full test coverage is provided.
  - **Alpha release (GOAL DATE: 11/26/23):** The application is able to run from an executable file across all systems specified in the CI. Bugs may be present, model might require optimization, documentation may be incomplete, baby removal via simple low-pass filtering.
  - **Beta release (Spring 2023 - not within scope of 487W):** Audio separation and removal via U-Net architecture, more training data, more complex model architecture for improved detection accuracy.
  - **Final release (Spring 2023 - not within scope of 487W):** Full documentation and public release.

### Team Structure
- **Caitlin Crowe:** Project manager & lead developer
- **(TODO: Israel's role)**
- **(TODO: Gwen's role)**
- **(TODO: Richard's role)**

### Documentation Plan
- **User Guide:** *README.md* - How to install and run the application.
- **Developer Guide:** *DEV_README.md (TODO: Create this after alpha release)* - How the application and its modules work.
- **API Documentation:** *Doxygen (TODO: Implement Doxygen)* - Documentation of the application's API.

### Testing Plan & CI

#### Test Automation
- **Infrastructure:** PyTest (https://docs.pytest.org/en/)
- **Justification:** PyTest is a popular, well documented, and easy to use framework for creating automated tests in Python, which comprises 100% of this project's code. It is also compatible with GitHub Actions, which is the CI infrastructure we are using.
- **Test Expansion:** To add new tests, simply expand upon the relevent test module within the `tests/` directory. To add a new test module, create a new file within the `tests/` directory with the name `test_<module_name>.py`. To add a new test function, create a new function within the relevent test module with the name `test_<function_name>`. To add a new test case, create a new assert statement within the relevant test function. Be sure to create relevant mock data and avoid overriding data files, such as test models and scalers, with files generated with mock data via `joblib.dump`.

#### Continuous Integration
- **Infrastructure:** GitHub Actions
- **Justification:** GitHub Actions is well documented, compatible with PyTest, and available directly from GitHub, which is where our code is hosted.
- ***(TODO: Pros/cons matrix for at least two CI services that you considered.)***
- **Tests to run:** All test modules should be run upon each CI build.
- **When to build:** CI builds should be run upon each push to the `main` branch, as well as upon each pull request to the `main` branch. *(TODO: Update workflow file to reflect this behavior.)*

#### Test Cases
- **Audio Classification Module**
  - **Tests:**
    - `test_audio_classifier_initiation`
      - **Purpose:** Validates the proper instantiation of the `AudioClassifier` class.
      - **Method:** Checks if an object created using the `AudioClassifier` constructor is indeed an instance of `AudioClassifier`.
    - `test_new_model_training`
      - **Purpose:** Tests the `new_model` method responsible for training a new model.
      - **Method:**
        - Generates mock features and labels for binary classification (both positive and negative classes).
        - Calls `new_model` with these mock inputs and additional parameters like `num_epochs` and `verbose`.
        - Verifies that the scaler is initialized and the model is saved after training. Also checks that the mock scaler is not saved.
    - `test_load_model`
      - **Purpose:** Ensures that the `load_model` method correctly loads a pre-trained model.
      - **Method:**
        - Mocks the `torch.load` function to return a predefined state dictionary mimicking a trained model.
        - Calls `load_model` and checks whether the scaler is also loaded alongside the model, ensuring the model's readiness for inference.
    - `test_model_evaluation`
      - **Purpose:** Validates the model's ability to perform a forward pass (evaluation) correctly.
      - **Method:**
        - Creates a mock input tensor.
        - Performs a forward pass using the `AudioClassifier` instance.
        - Checks if the output tensor has the correct shape, corresponding to the batch size and the expected output nodes (2 for binary classification).

- **Audio Processing Module**
  - **Tests:**
    - `test_audio_callback`
      - **Purpose:** Tests the `audio_callback` method of the `AudioProcessor` class to ensure it processes audio data correctly.
      - **Method:**
        - Instantiates `AudioProcessor` with mock model and data plotter.
        - Simulates a callback with random input audio data.
        - Validates that the model's evaluation method is called and that the plotter's update methods are invoked.
        - Checks that the output audio data is modified as expected, indicating processing has occurred.
    - `test_compute_mfcc`
      - **Purpose:** Verifies that the `compute_mfcc` function computes Mel-frequency cepstral coefficients (MFCCs) correctly.
      - **Method:**
        - Computes MFCCs using a mock audio buffer.
        - Asserts that the resulting MFCC array has the correct shape, matching the expected number of coefficients.
    - `test_normalize_audio_length`
      - **Purpose:** Ensures the `normalize_audio_length` function correctly pads short audio data or truncates long audio data to a target length.
      - **Method:**
        - Mocks `librosa.load` to return arrays of specific lengths (shorter and longer than the target audio length).
        - Uses the `normalize_audio_length` function to process both short and long mock audio data.
        - Confirms that the output audio data lengths match the desired target length, validating correct padding and truncation behavior.

- **Data Plotting Module**
  - **Tests:**
    - `test_audio_data_plotter_initialization`
      - **Purpose:** Tests the initialization of the `AudioDataPlotter` class to ensure it is set up correctly.
      - **Method:**
        - Initializes an `AudioDataPlotter` instance with given MFCC and range values.
        - Checks if the `n_mfcc`, `mfcc_range`, and shape of `mfccs` in the plotter are set correctly as per the inputs.
    - `test_update_mfcc_data`
      - **Purpose:** Verifies the `update_mfcc_data` method of the `AudioDataPlotter` class.
      - **Method:**
        - Creates an `AudioDataPlotter` instance and updates its MFCC data with a mock MFCC data array.
        - Asserts that the `mfccs` attribute of the plotter is updated to match the provided mock MFCC data.
    - `test_update_prediction_text`
      - **Purpose:** Tests the `update_prediction_text` method of the `AudioDataPlotter` class.
      - **Method:**
        - Initializes an `AudioDataPlotter` instance and sets its prediction text to a test string.
        - Validates that the `prediction_text` attribute of the plotter is updated to match the test string.
    - `test_update_filter_activity_text`
      - **Purpose:** Checks the `update_filter_activity_text` method of the `AudioDataPlotter` class.
      - **Method:**
        - Creates an `AudioDataPlotter` instance and updates its filter activity text with a test string and transparency value.
        - Confirms that the `filter_activity_text` attribute and the alpha (transparency) of the filter activity text element are updated as expected.

- **File Management Module**
  - **Tests:**
    - `test_load_data`
      - **Purpose:** Tests the `load_data` function, ensuring it correctly loads and processes audio files from a given directory.
      - **Method:**
        - Mocks `os.listdir` to return a list of filenames and `librosa.load` to return mock audio data.
        - Calls `load_data` with a mock folder path, label, and audio processing parameters.
        - Verifies that the number of features and labels returned matches the expected count.
        - Checks that each feature has the correct length and that all labels match the expected label.
    - `test_count_files_in_folder`
      - **Purpose:** Tests the `count_files_in_folder` function to ensure it accurately counts the number of files in a specified folder.
      - **Method:**
        - Mocks `os.path.exists` to return `True`, indicating the folder exists.
        - Mocks `os.listdir` to return a list of file names and `os.path.isfile` to treat each list item as a file.
        - Calls `count_files_in_folder` with a mock folder path.
        - Asserts that the count of files is as expected (in this case, 2).
    - `test_count_files_in_non_existing_folder`
      - **Purpose:** Checks how `count_files_in_folder` handles a non-existing folder.
      - **Method:**
        - Mocks `os.path.exists` to return `False`, indicating the folder does not exist.
        - Calls `count_files_in_folder` with a mock non-existing folder path.
        - Verifies that the function returns an appropriate response ("Folder not found.") for a non-existing folder.

- **GUI Module**
  - **Tests:**
    - `test_device_selector_initialization`
      - **Purpose:** Checks correct initialization of DeviceSelector class.
      - **Method:**
        - Mocks `sounddevice.query_devices` to return preset device data.
        - Validates that the `DeviceSelector` input and output device lists match expected values.
    - `test_confirm_button`
      - **Purpose:** Ensures correct functionality of the confirm button in DeviceSelector.
      - **Method:**
        - Mocks `sounddevice.query_devices`.
        - Simulates device selection in `DeviceSelector`.
        - Validates that input and output selections are accurately stored.

