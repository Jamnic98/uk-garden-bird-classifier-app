# UK Garden Bird Classifier

A custom convolutional neural network trained to recognise common UK garden birds, integrated with a web interface for predictions.

## Overview

This project explores building a CNN from scratch using **PyTorch** to classify common UK garden birds. The goal was to create a model that could generalise well beyond the training data and serve predictions through a responsive web interface.

## Dataset

- Collected images using a **custom web scraper** focused on UK garden species  
- Approximately **120 images per class**, producing a balanced dataset  
- Dataset designed for generalisation to real-world garden photos  

## Model Architecture

- Initial model: simple CNN with a few convolutional layers  
- Improvements:
  - Added depth with more convolutional layers  
  - Introduced **max pooling** for downsampling  
  - Added **dropout** layers to reduce overfitting  
- Real-time **data augmentation** via PyTorch DataLoader  
- Multiple workers to prevent GPU bottlenecks during training  

**Final model accuracy:** 97.8%, with strong validation results across all classes  

## Deployment

- Model exported to **ONNX** for portability  
- **FastAPI** service handles inference requests  
- Frontend built with **Jinja2**:
  - Images uploaded via interface
  - Encoded in Base64
  - Decoded and preprocessed into tensors for prediction
- Rate limiting and interface safeguards prevent backend overload  

## Tech Stack

- Python  
- PyTorch  
- ONNX  
- FastAPI  
- Jinja2  

## Outcome

This project demonstrates:
- Building and tuning a CNN from scratch  
- Implementing real-time data augmentation and GPU optimisations  
- Serving predictions via a robust web interface  
- Practical deployment considerations such as rate limiting and request handling  

## GitHub Repository

[GitHub Repo](https://github.com/Jamnic98/uk-garden-bird-classifier-app)
