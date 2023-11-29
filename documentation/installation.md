# BabyBash
*Installation & Usage*


## Table of Contents
- [Overview](../README.md)
- Installation & Usage
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
- [Architecture](architecture.md)
  - [Architecture Overview](architecture.md#architecture-overview)
  - [Software Components](architecture.md#software-components)
  - [Assumptions](architecture.md#assumptions)
  - [Alternative Architectures](architecture.md#alternative-architectures)
- [Design](design.md)
  - [Modules](design.md#modules)
  - [Coding Guidelines](design.md#coding-guidelines)
  - [GitHub Structure](design.md#github-structure)
  - [Process Description](design.md#process-description)
- [Testing & CI](testing.md)
  - [Test Automation](testing.md#test-automation)
  - [Continuous Integration](testing.md#continuous-integration)
  - [Test Cases](testing.md#test-cases)
- [Developer Guidelines](dev_guidelines.md)
  - *(TODO: Complete `dev_guidelines.md`)*


## Requirements
- **Apple Silicon machine**
- **[Git](https://git-scm.com/downloads)**
- ***Recommended:* [BlackHole](https://existential.audio/blackhole/)**
  - This is a virtual audio driver that allows for routing of computer audio output as input. Download this if you wish to filter computer audio using BabyBash.


## Installation
**BabyBash is currently only available on Apple Silicon machines.**<br>
*(NOTE: We have not yet tested on Linux, it may be possible to build on there. If you are successful in doing so, please contact Caitlin at `ckc5828@psu.edu`.)*

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
***(NOTE: Please read the warning below before running BabyBash.)***<br>
`python src/main.py`


## Usage
**WARNING: Be sure not to pick an input device that can hear your output to avoid *LOUD* feedback loops!**
<br><br>

After installing and running BabyBash, babies will be filtered out of the input audio stream, and the filtered audio will output to the selected output device. To stop filtering, simply close the program.
<br><br>

If you wish to filter computer audio, provided you have installed BlackHole (or a similar program), set your computer's output device to BlackHole (this prevents your "clean" computer audio from being audible alongside the filtered audio). Then, set the input device in BabyBash to BlackHole (this feeds your computer output into BabyBash). You can now play audio from your computer and it will be filtered by BabyBash!
