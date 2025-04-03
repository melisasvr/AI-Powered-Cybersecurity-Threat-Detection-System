# AI-Powered Cybersecurity Threat Detection System
This project implements an AI-powered system to detect cybersecurity threats across three domains: network anomalies, malware, and phishing emails. It uses machine learning models from scikit-learn, including Isolation Forest, Random Forest, and an MLP-based autoencoder, to identify potential threats in synthetic data. The system is designed to train models, test them, and save them for future use, making it a modular and extensible framework for cybersecurity analysis.

## Features
- Network Anomaly Detection: Identifies unusual network traffic patterns using an autoencoder and Isolation Forest.
- Malware Detection: Classifies executable files as benign or malicious based on simulated PE file features.
- Phishing Detection: Detects phishing emails using text analysis with TF-IDF and Random Forest.
- Synthetic Data Generation: Creates realistic test data for all three domains.
- Model Persistence: Saves trained models to disk for reuse.

## Prerequisites
- Python: Version 3.8 or higher (tested with Python 3.12).
- Dependencies: Install required libraries using the provided requirements.txt.

## Installation
1. Clone the Repository (if hosted on GitHub or similar):
- git clone https://github.com/yourusername/ai-cybersecurity-threat-detection.git
- cd ai-cybersecurity-threat-detection

2. Set Up a Virtual Environment (optional but recommended):
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies:
- Create a requirements.txt file with the following content:
```numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.2.0

Then run:
pip install -r requirements.txt
```

## Usage: 
1. Run the Main Script:
Execute main.py to train, test, and save the models:
- python main.py

- The script will:
- Train all models on synthetic data.
- Test them on sample data for network traffic, malware files, and phishing emails.
- Save trained models to the ./models directory.

2. Expected Output:
Training Cybersecurity Threat Detection System...
1. Training Network Anomaly Detection Models...
2. Training Malware Detection Model...
3. Training Phishing Detection Model...
```All models trained successfully!
===== TESTING THREAT DETECTION SYSTEM =====

1. Testing Network Anomaly Detection...
Network traffic analysis:
  - Total traffic flows: 1000
  - Detected anomalies: <number>
  - Anomaly percentage: <percentage>%

2. Testing Malware Detection...
Malware detection results:
  - benign_test_1.exe: <BENIGN/MALWARE> (confidence: <percentage>%)
  - malware_test_1.exe: <BENIGN/MALWARE> (confidence: <percentage>%)
  - suspicious_file.exe: <BENIGN/MALWARE> (confidence: <percentage>%)

3. Testing Phishing Detection...
Phishing detection results:
  - Email 1: <LEGITIMATE/PHISHING> (confidence: <percentage>%)
  - Email 2: <LEGITIMATE/PHISHING> (confidence: <percentage>%)
  - Email 3: <LEGITIMATE/PHISHING> (confidence: <percentage>%)

All models saved to ./models
===== SYSTEM READY FOR DEPLOYMENT =====
```

3. Custom Usage:
- Modify main.py to use real data instead of synthetic data by adjusting the input to the detector.detect_* methods.
- Load saved models using detector.load_models() and run predictions without retraining.


## File Structure
```AI-Powered-Cybersecurity-Threat-Detection/
├── main.py              # Main script with all functionality
├── models/              # Directory for saved models (created after running)
│   ├── network_autoencoder.pkl
│   ├── network_isoforest.pkl
│   ├── network_scaler.pkl
│   ├── malware_detector.pkl
│   ├── phishing_vectorizer.pkl
│   └── phishing_detector.pkl
├── requirements.txt     # List of Python dependencies
└── README.md            # This file
```

## How It Works
- Network Anomaly Detection:
- Uses an MLPRegressor as an autoencoder and Isolation Forest to detect anomalies.
- Features: bytes sent/received, duration, port, protocol type, service, and flag.
- Preprocessing: One-hot encoding for categorical features, scaling for numerical features.
- Malware Detection:
- Employs a Random Forest classifier on simulated PE file features (e.g., file size, entropy, imports).
- Labels: 0 (benign), 1 (malware).
- Phishing Detection:
- Uses TF-IDF vectorization and Random Forest to classify emails.
- Features: Text content analyzed for suspicious patterns (e.g., urgent language, misspelling).
- Synthetic Data:
- It generates realistic data for training and testing, seeded for reproducibility.

## Contributing
- You can fix this project, submit pull requests, or open issues for bugs and feature requests.

## License
- This project is unlicensed; use it freely for educational or personal purposes. For commercial use, don't hesitate to get in touch with the author.

## Author
- Your Name: Melisa Sever

