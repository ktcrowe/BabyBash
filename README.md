# BabyBash
AI Audio Removal

This is a prototype of AI driven software that aims to remove unwanted noises from input audio signals.  In this case, we aim to remove the sound of crying babies.

This project uses audio files from both the DonateACry audio corpus (https://github.com/gveres/donateacry-corpus) and the ESC-50 Dataset (https://github.com/karolpiczak/ESC-50) for both positive and negative audio samples.

Current branches:
  - interference: removal of the detected cry via destructive interference & output of the enhanced audio
  - optimization: improvements to the neural network to increase accuracy
