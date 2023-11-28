# BabyBash
AI Audio Removal

## Overview

This is a prototype of AI driven software that aims to remove unwanted noises from input audio signals.  In this case, we aim to dampen the sound of crying babies.

This project uses audio files from both the DonateACry audio corpus for positive audio samples (https://github.com/gveres/donateacry-corpus) and the ESC-50 Dataset for both positive and negative audio samples (https://github.com/karolpiczak/ESC-50).

### Current branches:
  - interference: dampening of the detected cry & output of the filtered audio
  - optimization: improvements to the neural network to increase accuracy
  - testing: test cases to ensure all functions are working as expected
  - refactor: code refactoring to improve readability & encapsulation
  - gui: development of a GUI to allow for toggling of the filter and choice of audio devices
  - documentation: incorporation of Doxygen documentation into the project
  - build: used for building the project with PyInstaller

*(Note: Ownership transferred from personal to school account on 11/20/2020 due to workflow limitations.)*

## Installation
BabyBash is currently only available on Apple Silicon machines.
*(Note: We have not yet tested on Linux, it may be possible to build on there. If you are successful in doing so, please contact Caitlin at `ckc5828@psu.edu`.)*

### Requirements
- **Git** (https://git-scm.com/downloads)
- **Recommended: BlackHole** (https://existential.audio/blackhole/)
  - This is a virtual audio driver that allows for routing of computer audio output as input. Download this if you wish to filter computer audio using BabyBash.

### 1. Download Miniconda
*(NOTE: If you already have Anaconda or Miniconda installed on your machine, you can skip this step.)*
<br><br>
Run the following commands in your terminal to download Miniconda:
```
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, be sure to initialize Miniconda with the following commands:
```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### 2. Verify Git and Miniconda Installations
Run the following command in your terminal to ensure Git is installed:<br>
`git --version`
<br><br>
Similarly, run the following command to ensure Miniconda is installed:<br>
`conda --version`
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
`cd BabyBash`

### 4. Create the Conda Environment
Run the following commands to create and activate the Conda environment:<br>
```
conda env create -f environment.yml
conda activate babybashenv
```

### 5. Run BabyBash
Now you can run the program! Simply run the following command:<br>
`python src/main.py`
<br><br>
Now babies will be filtered out of the input audio stream, and the filtered audio will output to the selected output device.
<br><br>
If you wish to filter computer audio, set your computer's output device to BlackHole (this prevents your "clean" computer audio from being audible alongside the filtered audio). Then, set the input device in BabyBash to BlackHole. You can now play audio from your computer and it will be filtered by BabyBash!
