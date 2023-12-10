# BabyBash
*AI driven sound removal.*


## Overview
This is a prototype of AI driven software that aims to block unwanted noises from input audio signals.  In this case, we aim to dampen the sound of crying babies from an input signal.

This prototype's training data consists of audio files from both the [DonateACry audio corpus](https://github.com/gveres/donateacry-corpus) for positive samples and the [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) for both positive and negative samples.

*(Note: Ownership transferred from personal to school account on 11/20/2020 due to workflow limitations.)*


## Requirements
- **Apple Silicon machine**
- **[Git](https://git-scm.com/downloads)**
- **[Poetry](https://python-poetry.org/docs/#installation)**
- ***Recommended:* [BlackHole](https://existential.audio/blackhole/)**
  - This is a virtual audio driver that allows for routing of computer audio output as input. Download this if you wish to filter computer audio using BabyBash.


## Installation
**BabyBash is currently only available on Apple Silicon machines.**<br>

### 1. Download Poetry
Run the following command in your terminal to download Poetry:
```
pip install poetry
```

### 2. Verify Git and Poetry Installations
Run the following command in your terminal to ensure Git is installed:<br>
`git --version`
<br><br>
Similarly, run the following command to ensure Poetry is installed:<br>
`poetry --version`
<br><br>
If both of these commands return a version number, you are good to go!

### 3. Clone the BabyBash Repository
Begin by navigating to the directory you wish to clone the repository into.<br>
In the following example, BabyBash will be installed into the Downloads folder:<br>
`cd ~/Downloads`
<br><br>
Next, clone the repository to this folder:<br>
`git clone https://github.com/caitcrowe/BabyBash`
<br><br>
After cloning the repository, navigate into the BabyBash directory:<br>
`cd BabyBash`<br><br>
*(Note: You can safely delete the `data` directory in the cloned repository to save disk space if you do not plan on training new models.)*

### 4. Create the Poetry Environment
Run the following commands to create and activate the Poetry environment:<br>
`poetry install`<br>
`poetry shell`

### 5. Run BabyBash
Now you can run the program! Simply run the following command:<br>
***(Note: Please read the warning below before running BabyBash.)***<br>
`python babybash`


## Usage
**WARNING: Be sure not to pick an input device that can hear your output to avoid *LOUD* feedback loops!**
<br><br>
After installing and running BabyBash, babies will be filtered out of the input audio stream, and the filtered audio will output to the selected output device. To stop filtering, simply close the program.
<br><br>
If you wish to filter computer audio, provided you have installed BlackHole (or a similar program), set your computer's output device to BlackHole (this prevents your "clean" computer audio from being audible alongside the filtered audio). Then, set the input device in BabyBash to BlackHole (this feeds your computer output into BabyBash). You can now play audio from your computer and it will be filtered by BabyBash!


## Documentation
### Table of Contents
- Overview
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
- [Architecture](documentation/architecture.md)
  - [Architecture Overview](documentation/architecture.md#architecture-overview)
  - [Software Components](documentation/architecture.md#software-components)
  - [Assumptions](documentation/architecture.md#assumptions)
  - [Alternative Architectures](documentation/architecture.md#alternative-architectures)
- [Design](documentation/design.md)
  - [Modules](documentation/design.md#modules)
  - [Coding Guidelines](documentation/design.md#coding-guidelines)
  - [Project Structure](documentation/design.md#project-structure)
  - [GitHub Structure](documentation/design.md#github-structure)
  - [Process Description](documentation/design.md#process-description)
- [Testing & CI](documentation/testing.md)
  - [Test Automation](documentation/testing.md#test-automation)
  - [Continuous Integration](documentation/testing.md#continuous-integration)
  - [Test Cases](documentation/testing.md#test-cases)
- [Developer Guidelines](documentation/dev_guidelines.md)
  - *(TODO: Complete `dev_guidelines.md`)*