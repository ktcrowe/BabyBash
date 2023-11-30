# BabyBash
*Testing & CI*


### Table of Contents
- [Overview](../README.md)
  - [Requirements](../README.md#requirements)
  - [Installation](../README.md#installation)
  - [Usage](../README.md#usage)
- [Architecture](architecture.md)
  - [Architecture Overview](architecture.md#architecture-overview)
  - [Software Components](architecture.md#software-components)
  - [Assumptions](architecture.md#assumptions)
  - [Alternative Architectures](architecture.md#alternative-architectures)
- [Design](design.md)
  - [Modules](design.md#modules)
  - [Coding Guidelines](design.md#coding-guidelines)
  - [Project Structure](design.md#project-structure)
  - [GitHub Structure](design.md#github-structure)
  - [Process Description](design.md#process-description)
- Testing & CI
  - [Test Automation](#test-automation)
  - [Continuous Integration](#continuous-integration)
  - [Test Cases](#test-cases)
- [Developer Guidelines](dev_guidelines.md)
  - *(TODO: Complete `dev_guidelines.md`)*


## Test Automation
- **Infrastructure:** [PyTest](https://docs.pytest.org/en/)
- **Justification:** PyTest is a popular, well documented, and easy to use framework for creating automated tests in Python, which comprises 100% of this project's code. It is also compatible with GitHub Actions, which is the CI infrastructure we are using.
- **Test Expansion:** To add new tests, simply expand upon the relevant test module within the `tests/` directory. To add a new test module, create a new file within the `tests/` directory with the name `test_<module_name>.py`. To add a new test function, create a new function within the relevant test module with the name `test_<function_name>`. To add a new test case, create a new assert statement within the relevant test function. Be sure to create relevant mock data and avoid overriding data files, such as test models and scalers, with files generated with mock data via `joblib.dump`.

## Continuous Integration
- **Infrastructure:** GitHub Actions
- **Justification:** GitHub Actions is well documented, compatible with PyTest, and available directly from GitHub, which is where our code is hosted.
- ***(TODO: Pros/cons matrix for at least two CI services that you considered.)***
- **Tests to run:** All test modules should be run upon each CI build.
- **When to build:** CI builds should be run upon each push to the `main` branch, as well as upon each pull request to the `main` branch.

## Test Cases
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
