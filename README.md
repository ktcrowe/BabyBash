# BabyBash
*AI driven sound removal.*


## Table of Contents
- Overview
- [Installation & Usage](documentation/installation.md)
  - [Requirements](documentation/installation.md#requirements)
  - [Installation](documentation/installation.md#installation)
  - [Usage](documentation/installation.md#usage)
- [Architecture](documentation/architecture.md)
  - [Architecture Overview](documentation/architecture.md#architecture-overview)
  - [Software Components](documentation/architecture.md#software-components)
  - [Assumptions](documentation/architecture.md#assumptions)
  - [Alternative Architectures](documentation/architecture.md#alternative-architectures)
- [Design](documentation/design.md)
  - [Modules](documentation/design.md#modules)
  - [Coding Guidelines](documentation/design.md#coding-guidelines)
  - [GitHub Structure](documentation/design.md#github-structure)
  - [Process Description](documentation/design.md#process-description)
- [Testing & CI](documentation/testing.md)
  - [Test Automation](documentation/testing.md#test-automation)
  - [Continuous Integration](documentation/testing.md#continuous-integration)
  - [Test Cases](documentation/testing.md#test-cases)
- [Developer Guidelines](documentation/dev_guidelines.md)
  - *(TODO: Complete `dev_guidelines.md`)*


## Overview
This is a prototype of AI driven software that aims to block unwanted noises from input audio signals.  In this case, we aim to dampen the sound of crying babies from an input signal.

This prototype's training data consists of audio files from both the [DonateACry audio corpus](https://github.com/gveres/donateacry-corpus) for positive samples and the [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50) for both positive and negative samples.

*(Note: Ownership transferred from personal to school account on 11/20/2020 due to workflow limitations.)*
