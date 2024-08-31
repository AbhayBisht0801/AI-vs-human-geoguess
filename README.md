

# GeoGuessr-Like Game: AI vs Human

Welcome to my GeoGuessr-inspired project! This application allows you to explore the world through images and test your skills against an AI model or play solo. The AI, a CNN model trained on 10,000 images, predicts the longitude and latitude of the images, challenging you to see who can guess more accurately.

## Features
- **AI vs Human Mode**: Compete against a CNN model that predicts the geographical coordinates based on the provided images.
- **Solo Mode**: Play solo and test your skills in guessing the location of an image.
- **Modular Codebase**: The project is built with modular code, making it easy to understand, extend, and maintain.
- **Local Deployment**: The entire application is deployed locally using Flask.

## How It Works
1. **AI Model**: A Convolutional Neural Network (CNN) trained on a dataset of 10,000 images predicts the longitude and latitude.
2. **Game Modes**: Choose between AI vs Human or Solo mode to start guessing locations based on the provided images.
3. **Flask Application**: The game is hosted locally using Flask, providing a smooth and interactive experience.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AbhayBisht0801/AI-vs-human-geoguess.git
   cd AI-vs-human-geoguess
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Dataset**
   Run the data ingestion script to set up the dataset:
   ```bash
   python pipeline/data_ingestion.py
   ```

4. **Run the Flask Application**
   ```bash
   python app.py
   ```

   Open your browser and navigate to `http://127.0.0.1:5000` to start playing.

## Tech Stack
- **Python**
- **Flask**
- **CNN Model (TensorFlow/Keras)**
- **HTML/CSS/JavaScript** for the frontend

