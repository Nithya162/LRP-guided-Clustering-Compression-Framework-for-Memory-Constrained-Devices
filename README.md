# HiPC25 Project

A Layer-wise Adaptive Precision Assignment and Compression framework using Layer-wise Relevance Propagation (LRP) driven by clustering for neural network compression.

## Project Structure

## Summary

To run the entire pipeline with this structure:

## Usage

Run from the project root directory:

```bash
# Run all phases for both models
python main.py --phase all --model both

# Run specific phase (1, 2, or 3) for a specific model (fc or resnet)
python main.py --phase 1 --model fc

# Run with custom device (cuda or cpu)
python main.py --phase all --model both --device cuda

# Get help with command line options
python main.py --help
