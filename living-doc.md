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
*(TODO: Update to reflect GUI class properly after implementation.)*
- **Classes:** `DeviceSelector`
- **Functions:** `handle_toggle_filter`
- **Responsibilities:** 
  - `DeviceSelector`: Allow user to choose input and output devices on program startup.
  - `handle_toggle_filter`: Allow for the toggling of the filter on and off.


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
