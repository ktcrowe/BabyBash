# BabyBash
*Design*


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
- Design
  - [Modules](#modules)
  - [Coding Guidelines](#coding-guidelines)
  - [Project Structure](#project-structure)
  - [GitHub Structure](#github-structure)
  - [Process Description](#process-description)
- [Testing & CI](testing.md)
  - [Test Automation](testing.md#test-automation)
  - [Continuous Integration](testing.md#continuous-integration)
  - [Test Cases](testing.md#test-cases)
- [Developer Guidelines](dev_guidelines.md)
  - *(TODO: Complete `dev_guidelines.md`)*


## Modules
*(TODO: Include attributes.)*
### Audio Classification Module
- **Classes:** `AudioClassifier`
- **Responsibilities:** 
  - `AudioClassifier`: Load existing models, train new models if needed, classify input audio data in real-time.
    - **Parameters:**
      - `n_mfcc`: The number of MFCC coefficients to compute when classifying audio data.
      - `models_path`: The path to the folder containing the model files.
    - **Methods:**
      - `forward`: Preform a forward pass of the input data through the model.
        - **Parameters:**
          - `x`: The input data.
        - **Returns:** The output of the model.
      - `new_model`: Trains and saves a new model.
        - **Parameters:**
          - `positive_features`: The MFCC features of the positive training data.
          - `negative_features`: The MFCC features of the negative training data.
          - `positive_labels`: The labels of the positive training data.
          - `negative_labels`: The labels of the negative training data.
          - `test_size` *(Defaults to `0.2`)*: The proportion of the data to use for testing.
          - `random_state` *(Defaults to `42`)*: The random state to use for the train-test split.
          - `num_epochs` *(Defaults to `10`)*: The number of epochs to train the model for.
          - `verbose` *(Defaults to `False`)*: Whether to print verbose output to the console.
          - `save_scaler` *(Defaults to `True`)*: Whether to save the scaler used to normalize the training data.
      - `load_model`: Load a pre-trained model.
        - **Parameters:**
          - `model_name`: The file name of the model to load from `models_path`.
          - `verbose` *(Defaults to `False`)*: Whether to print verbose output to the console.

### Audio Processing Module
- **Classes:** `AudioProcessor`
- **Functions:** `compute_mfccs`, `initialize_filter`
- **Responsibilities:** 
  - `AudioProcessor`: Capture input audio, apply filter if needed, output filtered audio. Tells data plotter to update the animation accordingly.
    - **Parameters:**
      - `model`: The file name for the model to use for classification.
      - `data_plotter`: The instance of `DataPlotter` to update.
      - `sample_rate`: The sample rate of the audio signal.
      - `block_size`: The number of samples per block of audio data.
      - `n_mfcc`: The number of MFCC coefficients to compute when classifying audio data.
      - `n_fft`: The number of samples per fast Fourier-transform.
      - `hop_length`: The number of samples between successive frames.
      - `input_device`: The input device to use.
      - `output_device`: The output device to use.
      - `verbose` *(Defaults to `False`)*: Whether to print verbose output to the console.
    - **Methods:**
      - `audio_callback`: The callback function to be called when new audio data is available.
        - **Parameters:**
          - `indata`: The new audio data.
          - `outdata`: The audio data to output.
          - `frames`: The number of frames in the audio data.
          - `time`: The time of the audio data.
          - `status`: The status of the audio data.
      - `update_detection_buffer`: Removes the oldest classification from the buffer and adds the newest classification to the buffer.
        - **Parameters:**
          - `current_detection`: The newest classification to add to the buffer.
      - `calculate_fade_factor`: Calculate the fade factor to apply to the audio signal filter.
      - `update_filter_activity_text`: Update the text indicating whether the filter is being applied or not.
        - **Parameters:**
          - `fade_factor`: The transparency of the text corresponding to the filter's fade factor.
      - `open_streams`: Open the input and output audio streams.
  - `compute_mfcc`: Compute the MFCC features of a given audio signal.
    - **Parameters:**
      - `indata`: The audio signal to compute the MFCC features of.
      - `sample_rate`: The sample rate of the audio signal.
      - `n_mfcc`: The number of MFCC coefficients to compute.
      - `n_fft`: The number of samples per fast Fourier-transform.
      - `hop_length`: The number of samples between successive frames.
    - **Returns:** The mean of the MFCC features of the audio signal.
  - `initialize_low_pass_filter`: Initialize a low-pass filter to be applied to the audio signal.
    - **Parameters:**
      - `cutoff_frequency`: The cutoff frequency of the filter. Frequencies above this value will be filtered out.
      - `sample_rate`: The sample rate of the audio signal.
      - `order` *(Defaults to `5`)*: The order of the filter. Higher orders will result in a steeper cutoff.
    - **Returns:** A tuple containing the two filter coefficients.
  - `normalize_audio_length`: Normalize the length of an audio signal to a given length.
    - **Parameters:**
      - `audio_path`: The path to the audio file to normalize.
      - `target_length`: The length to normalize the audio file to.
      - `sample_rate`: The sample rate of the audio file.
    - **Returns:** The audio file normalized to the given length.

### Data Plotting Module
- **Classes:** `DataPlotter`
- **Responsibilities:** 
  - `DataPlotter`: Plot input MFCC data, animate in real-time, update relative information when needed (e.g. "Baby crying detected!").
    - **Parameters:**
      - `n_mfcc`: The number of MFCC coefficients to compute when plotting the input signal.
      - `mfcc_range`: The range of MFCC coefficient values to draw on the y-axis of the plot. The y-axis will range from `-mfcc_range` to `mfcc_range`.
    - **Methods:**
      - `update_mfcc_values`: Update the MFCC values to be plotted.
        - **Parameters:**
          - `mfcc_values`: The new MFCC values to plot.
      - `update_prediction_text`: Update the text indicating whether a baby is crying or not.
        - **Parameters:**
          - `text`: The new text to display.
      - `update_filter_activity_text`: Update the text indicating whether the filter is being applied or not.
        - **Parameters:**
          - `text`: The new text to display.
          - `transparency`: The transparency of the text.
      - `update_plot`:
        - **Parameters:**
          - `frame`: The next frame of the animation.
        - **Returns:** A tuple containing the update y-axis data and the two updated text elements.
      - `animate`: Define and run the plot animation.
        - **Returns:** The defined animation object.

### File Management Module
- **Functions:** `load_data`, `count_files`
- **Responsibilities:** 
  - `load_data`: Load MFCC data from files implementing the audio processing module to do so.
    - **Parameters:**
      - `folder`: The path to the folder containing the audio data files.
      - `label`: The label to assign to the data.
      - `sample_rate`: The sample rate of the audio data.
      - `n_mfcc`: The number of MFCC coefficients to compute when processing the audio data.
      - `normalize_to_length` *(Defaults to `None`)*: The length to normalize the audio data to. If `None`, no normalization is performed.
    - **Returns:** A tuple containing the MFCC data and the label.
  - `count_files_in_folder`: Count the number of files in a given directory for naming new models appropriately.
    - **Parameters:**
      - `folder_path`: The path to the folder to count files in.
    - **Returns:** The number of files in the given folder path.

### GUI Module
- **Classes:** `DeviceSelector`
- **Responsibilities:** 
  - `DeviceSelector`: Build the GUI to prompt users to pick audio devices on program startup.
    - **Methods:**
      - `create_widgets`: Create and draw the GUI elements.
      - `confirm`: Confirm the selected devices and close the GUI.
      - `select_devices`: Run the GUI and return the selected devices.


## Coding Guidelines

### Overview
- **Language:** Python
- **Style Guide:** [Flake8](https://pypi.org/project/flake8/)
- **Explanation:** *Flake8 is a wrapper for [pyflakes](https://pypi.org/project/pyflakes/) and [PEP 8](https://peps.python.org/pep-0008/) that is the current standard for Python projects.*


## Project Structure
  - `root/`: Contains README, Anaconda environment data, and the pytest configuration file.
    - `.github/workflow/`: Contains the .yml file for the GitHub Actions CI.
    - `.idea/`: Contains configuration for IntelliJ IDEA.
    - `data/`: Contains the training data for the audio classification model.
    - `documentation/`: Contains all documentation markdown files.
    - `models/`: Contains the existing audio classification models and scaler.
    - `babybash/`: Contains all [modules](#modules) outlined in this file (`design.md`).
    - `tests/`: Contains all PyTest [tests](testing.md#test-cases) outlined in `testing.md`.


## GitHub Structure
### Branches:
  - **interference:** dampening of the detected cry & output of the filtered audio
  - **optimization:** improvements to the neural network to increase accuracy
  - **testing:** test cases to ensure all functions are working as expected
  - **refactor:** code refactoring to improve readability & encapsulation
  - **gui:** development of GUI elements
  - **documentation:** documentation of the project


## Process Description

### Risk Assessment
- **Risk:** Separation of baby crying from other sounds is too complex to be achieved in real-time.
  - **Likelihood:** High
  - **Impact:** High
  - **Evidence:** Since Deezer's "spleeter" library is not available to us, we would have to implement our own second NN using something akin to U-Net architecture. This would be a very complex task that is likely to not be feasible in the time we have.
  - **Reduction:** Experimented with different approaches to the problem of audio separation.
  - **Detection:** Audio can not be separated in real-time, or at all.
  - **Mitigation:** Apply a simple filter to the audio signal to muffle the sound of the baby crying.

- **Risk:** Detection of baby crying is not accurate enough to provide full coverage without false positives.
  - **Likelihood:** High
  - **Impact:** High
  - **Evidence:** The model is trained on a small dataset of only 1000 samples. This is not enough to provide full coverage of all possible baby crying sounds. The real-time nature of the application means that detections will be less accurate.
  - **Reduction:** Experimented with different approaches to the problem of audio classification, such as CNN architecture. This was not feasible given time constraints and lack of time axis in real-time audio data. Experimented with audio file length normalization and different amounts of MFCC coefficients to no avail.
  - **Detection:** The model is not able to detect baby crying in some cases. In other cases, the model detects baby crying when there is none.
  - **Mitigation:** Make use of fades and audio buffers to increase the window of audio filtering (filling in the blanks so to speak). Implement a threshold in the buffer to reduce false-positive filtering. Include more training data in the model.

### Project Schedule
- **Milestones:** Prototype completion, Alpha release, Beta release, Final release
  - **Prototype completion *(REACHED 11/24/23):*** Basic functionality of the application is implemented. The audio processing module is able to capture input audio and output filtered audio. The audio classification module is able to classify audio data as either baby crying or not baby crying. The data plotting module is able to plot input audio data in real-time. The file management module is able to load data from files. The GUI module is able to allow the user to choose input and output devices and toggle the filter. Full test coverage is provided.
  - **Alpha release *(REACHED 11/27/23):*** Partial documentation. The application is able to run via terminal across all systems specified in the CI. CI may be incomplete, bugs may be present, model might require optimization, baby removal via simple low-pass filtering.
  - **Beta release *(Spring 2024):*** Full documentation. True removal of unwanted noise from input signal. Signal delay may be present. Pivoting away from crying babies to simpler mechanical sound.
  - **Final release *(2024):*** Fully optimized model and real-time performance.

### Team Structure
- **Caitlin Crowe:** Project management & lead development
- **Bryan Naroditskiy:** Research assistance, data collection, and development

### Documentation Plan
**BabyBash** will be comprehensively documented via a series of markdown files in the repository, including but not limited to:
- **[User Guide](installation.md)**: `installation.md` - How to install, run, and use the application.
- **[Developer Guide](dev_guidelines.md)**: `dev_guidelines.md` - How to further develop the application.
- **[API Documentation](design.md#modules)**: *included within `design.md`* - Documentation of the application's classes and functions and how to use them.
