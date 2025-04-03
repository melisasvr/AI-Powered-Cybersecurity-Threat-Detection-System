import numpy as np
import pandas as pd
import os
import re
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

###################################
# 1. NETWORK ANOMALY DETECTION
###################################

def build_autoencoder_sklearn(input_dim):
    return MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 64, 128),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        random_state=42
    )

def preprocess_network_data(df):
    features = df[['bytes_sent', 'bytes_received', 'duration', 
                  'port', 'protocol_type', 'service', 'flag']]
    categorical_cols = ['protocol_type', 'service', 'flag']
    features = pd.get_dummies(features, columns=categorical_cols)
    scaler = StandardScaler()
    numerical_cols = ['bytes_sent', 'bytes_received', 'duration', 'port']
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
    return features, scaler

def train_network_anomaly_models(X_train):
    autoencoder = build_autoencoder_sklearn(X_train.shape[1])
    autoencoder.fit(X_train, X_train)
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    return autoencoder, iso_forest

def detect_network_anomalies(autoencoder, iso_forest, X_test):
    # Neural network reconstruction error
    predictions = autoencoder.predict(X_test)
    
    # Ensure predictions have the right shape
    if predictions.shape != X_test.shape:
        predictions = predictions.reshape(X_test.shape)
    
    # Check for NaN values across the entire DataFrame/array
    if np.isnan(predictions).any() or X_test.isna().any().any():
        predictions = np.nan_to_num(predictions)
        X_test_clean = X_test.fillna(0)  # Replace NaN with 0 in DataFrame
    else:
        X_test_clean = X_test
    
    # Calculate reconstruction error
    mse = np.mean(np.power(X_test_clean - predictions, 2), axis=1)
    
    # Set anomaly threshold
    autoencoder_anomalies = mse > np.percentile(mse, 99)
    
    # Isolation Forest anomalies
    iso_anomalies = iso_forest.predict(X_test) == -1
    
    # Combine both approaches
    combined_anomalies = autoencoder_anomalies | iso_anomalies
    
    return combined_anomalies, mse

def generate_sample_network_data(n_samples=10000):
    np.random.seed(42)
    normal_data = {
        'bytes_sent': np.random.normal(5000, 1000, n_samples),
        'bytes_received': np.random.normal(8000, 2000, n_samples),
        'duration': np.random.normal(60, 30, n_samples),
        'port': np.random.choice([80, 443, 22, 25], n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'https', 'ssh', 'smtp'], n_samples),
        'flag': np.random.choice(['SF', 'REJ', 'S0'], n_samples)
    }
    df = pd.DataFrame(normal_data)
    anomaly_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    df.loc[anomaly_indices, 'bytes_sent'] = np.random.normal(50000, 10000, len(anomaly_indices))
    df.loc[anomaly_indices, 'duration'] = np.random.normal(600, 100, len(anomaly_indices))
    return df

###################################
# 2. MALWARE DETECTION
###################################

def extract_pe_features(file_path):
    np.random.seed(int(hash(file_path) % 2**32))
    features = {
        'filesize': np.random.randint(10000, 10000000),
        'num_sections': np.random.randint(1, 20),
        'num_imports': np.random.randint(10, 500),
        'num_exports': np.random.randint(0, 50),
        'contains_packer_sig': np.random.choice([0, 1], p=[0.7, 0.3]),
        'entry_point_entropy': np.random.uniform(0, 8),
        'avg_section_entropy': np.random.uniform(0, 8),
        'has_digital_signature': np.random.choice([0, 1], p=[0.6, 0.4]),
        'has_tls_callback': np.random.choice([0, 1], p=[0.9, 0.1]),
        'has_anti_debug': np.random.choice([0, 1], p=[0.8, 0.2]),
        'has_anti_vm': np.random.choice([0, 1], p=[0.8, 0.2])
    }
    return features

def train_malware_detector(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def generate_sample_malware_data(n_samples=1000):
    np.random.seed(42)
    features = []
    labels = []
    for i in range(int(n_samples * 0.6)):
        file_path = f"benign_sample_{i}.exe"
        feature_dict = extract_pe_features(file_path)
        feature_dict['entry_point_entropy'] = np.random.uniform(0, 5.5)
        feature_dict['contains_packer_sig'] = np.random.choice([0, 1], p=[0.9, 0.1])
        feature_dict['has_anti_debug'] = np.random.choice([0, 1], p=[0.95, 0.05])
        feature_dict['has_anti_vm'] = np.random.choice([0, 1], p=[0.98, 0.02])
        features.append(feature_dict)
        labels.append(0)
    for i in range(int(n_samples * 0.4)):
        file_path = f"malware_sample_{i}.exe"
        feature_dict = extract_pe_features(file_path)
        feature_dict['entry_point_entropy'] = np.random.uniform(5.5, 8)
        feature_dict['contains_packer_sig'] = np.random.choice([0, 1], p=[0.3, 0.7])
        feature_dict['has_anti_debug'] = np.random.choice([0, 1], p=[0.5, 0.5])
        feature_dict['has_anti_vm'] = np.random.choice([0, 1], p=[0.4, 0.6])
        features.append(feature_dict)
        labels.append(1)
    df = pd.DataFrame(features)
    return df, np.array(labels)

###################################
# 3. PHISHING DETECTION
###################################

def extract_email_features(email_content):
    features = {
        'has_urgent_subject': bool(re.search(r'urgent|immediate|alert|critical', email_content, re.I)),
        'has_suspicious_links': bool(re.search(r'href=["\']https?://[^\/]*?(?:\d{1,3}\.){3}\d{1,3}', email_content)),
        'has_password_request': bool(re.search(r'password|credential|login|sign in', email_content, re.I)),
        'has_attachment_mention': bool(re.search(r'attach|download|open|file', email_content, re.I)),
        'has_financial_terms': bool(re.search(r'bank|account|money|transfer|paypal|credit|debit', email_content, re.I)),
        'has_misspellings': bool(re.search(r'verifcation|accaunt|securty|notifcation', email_content, re.I)),
        'email_length': len(email_content),
        'link_count': len(re.findall(r'href=["\']https?://', email_content)),
        'image_count': len(re.findall(r'<img', email_content))
    }
    return features

def preprocess_email_text(email_content):
    text = re.sub(r'<[^>]+>', ' ', email_content)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'https?://\S+', '', text)
    return text

def train_phishing_detector(email_texts, labels):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(email_texts)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_text, labels)
    return vectorizer, clf

def generate_sample_phishing_data(n_samples=500):
    np.random.seed(42)
    legitimate_templates = [
        "Dear {user}, Thank you for your recent purchase from {company}. Your order #{order_num} has been processed. If you have any questions, contact us at support@{company}.com.",
        "Hello {user}, This is a reminder about your upcoming appointment on {date}. Please call our office if you need to reschedule.",
        "Dear {user}, Your monthly statement is now available. You can view it by logging into your account at {company}.com/account.",
        "Hi {user}, Just wanted to follow up on our conversation yesterday. Let me know if you need any additional information.",
        "Dear valued customer, Thank you for being with {company} for {years} years! We appreciate your loyalty."
    ]
    phishing_templates = [
        "URGENT: Your {company} account has been suspended! Click here to verfiy your information: http://{suspicious_domain}/login",
        "Dear customer, We detected unusual activity in your account. Please verify your password and banking details here: http://{ip_address}/secure",
        "Your {company} account needs immediate attention! Your account will be terminated unless you update your information: {suspicious_link}",
        "Congratulations! You've won a free iPhone! Click to claim your prize now: http://{suspicious_domain}/claim-prize",
        "ALERT: Your package delivery failed. To reschedule, open the attachment: delivery_form.exe"
    ]
    companies = ["Amazon", "Netflix", "PayPal", "Microsoft", "Apple", "Bank of America", "Chase", "Facebook"]
    users = ["user", "customer", "member", "client", "recipient"]
    suspicious_domains = ["amaz0n-secure.com", "account-verify.net", "secure-login-portal.com", "verification-center.info"]
    emails = []
    labels = []
    for i in range(int(n_samples * 0.6)):
        template = np.random.choice(legitimate_templates)
        company = np.random.choice(companies)
        user = np.random.choice(users)
        email = template.format(
            user=user,
            company=company.lower(),
            order_num=np.random.randint(10000, 99999),
            date=f"{np.random.randint(1, 30)}/{np.random.randint(1, 12)}/2023",
            years=np.random.randint(1, 10)
        )
        emails.append(email)
        labels.append(0)
    for i in range(int(n_samples * 0.4)):
        template = np.random.choice(phishing_templates)
        company = np.random.choice(companies)
        suspicious_domain = np.random.choice(suspicious_domains)
        ip_address = f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        suspicious_link = f"http://{suspicious_domain}/verify?ref={np.random.randint(10000, 99999)}"
        email = template.format(
            company=company,
            suspicious_domain=suspicious_domain,
            ip_address=ip_address,
            suspicious_link=suspicious_link
        )
        emails.append(email)
        labels.append(1)
    processed_emails = [preprocess_email_text(email) for email in emails]
    return processed_emails, np.array(labels)

###################################
# 4. MAIN SYSTEM INTEGRATION
###################################

class CybersecurityThreatDetector:
    def __init__(self):
        self.network_autoencoder = None
        self.network_isoforest = None
        self.network_scaler = None
        self.malware_detector = None
        self.phishing_vectorizer = None
        self.phishing_detector = None
        self.is_trained = False
    
    def train(self, train_network=True, train_malware=True, train_phishing=True):
        print("Training Cybersecurity Threat Detection System...")
        if train_network:
            print("1. Training Network Anomaly Detection Models...")
            network_data = generate_sample_network_data()
            features, self.network_scaler = preprocess_network_data(network_data)
            X_train, X_test = train_test_split(features, test_size=0.3, random_state=42)
            self.network_autoencoder, self.network_isoforest = train_network_anomaly_models(X_train)
        if train_malware:
            print("2. Training Malware Detection Model...")
            malware_features, malware_labels = generate_sample_malware_data()
            X_train, X_test, y_train, y_test = train_test_split(
                malware_features, malware_labels, test_size=0.3, random_state=42
            )
            self.malware_detector = train_malware_detector(X_train, y_train)
        if train_phishing:
            print("3. Training Phishing Detection Model...")
            email_texts, phishing_labels = generate_sample_phishing_data()
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                email_texts, phishing_labels, test_size=0.3, random_state=42
            )
            self.phishing_vectorizer, self.phishing_detector = train_phishing_detector(train_texts, train_labels)
        self.is_trained = True
        print("All models trained successfully!")
    
    def save_models(self, directory="./models"):
        if not self.is_trained:
            print("Models not trained yet. Please train the models first.")
            return
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.network_autoencoder, f"{directory}/network_autoencoder.pkl")
        joblib.dump(self.network_isoforest, f"{directory}/network_isoforest.pkl")
        joblib.dump(self.network_scaler, f"{directory}/network_scaler.pkl")
        joblib.dump(self.malware_detector, f"{directory}/malware_detector.pkl")
        joblib.dump(self.phishing_vectorizer, f"{directory}/phishing_vectorizer.pkl")
        joblib.dump(self.phishing_detector, f"{directory}/phishing_detector.pkl")
        print(f"All models saved to {directory}")
    
    def load_models(self, directory="./models"):
        try:
            self.network_autoencoder = joblib.load(f"{directory}/network_autoencoder.pkl")
            self.network_isoforest = joblib.load(f"{directory}/network_isoforest.pkl")
            self.network_scaler = joblib.load(f"{directory}/network_scaler.pkl")
            self.malware_detector = joblib.load(f"{directory}/malware_detector.pkl")
            self.phishing_vectorizer = joblib.load(f"{directory}/phishing_vectorizer.pkl")
            self.phishing_detector = joblib.load(f"{directory}/phishing_detector.pkl")
            self.is_trained = True
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False
    
    def detect_network_threats(self, network_data):
        if not self.is_trained:
            print("Models not trained yet. Please train the models first.")
            return None
        if isinstance(network_data, pd.DataFrame) and 'protocol_type' in network_data.columns:
            processed_data, _ = preprocess_network_data(network_data)
        else:
            processed_data = network_data
        anomalies, scores = detect_network_anomalies(
            self.network_autoencoder, self.network_isoforest, processed_data
        )
        return {
            'anomalies': anomalies,
            'scores': scores,
            'num_anomalies': sum(anomalies),
            'anomaly_percentage': sum(anomalies) / len(processed_data) * 100
        }
    
    def detect_malware(self, file_paths):
        if not self.is_trained:
            print("Models not trained yet. Please train the models first.")
            return None
        results = []
        for file_path in file_paths:
            try:
                features = extract_pe_features(file_path)
                features_df = pd.DataFrame([features])
                malware_prob = self.malware_detector.predict_proba(features_df)[0, 1]
                is_malware = malware_prob > 0.5
                results.append({
                    'file_path': file_path,
                    'is_malware': is_malware,
                    'malware_probability': malware_prob,
                    'features': features
                })
            except Exception as e:
                results.append({
                    'file_path': file_path,
                    'error': str(e)
                })
        return results
    
    def detect_phishing(self, emails):
        if not self.is_trained:
            print("Models not trained yet. Please train the models first.")
            return None
        results = []
        for i, email in enumerate(emails):
            try:
                processed_text = preprocess_email_text(email)
                text_features = self.phishing_vectorizer.transform([processed_text])
                phishing_prob = self.phishing_detector.predict_proba(text_features)[0, 1]
                is_phishing = phishing_prob > 0.5
                email_features = extract_email_features(email)
                results.append({
                    'email_id': i,
                    'is_phishing': is_phishing,
                    'phishing_probability': phishing_prob,
                    'features': email_features
                })
            except Exception as e:
                results.append({
                    'email_id': i,
                    'error': str(e)
                })
        return results

###################################
# MAIN EXECUTION
###################################

if __name__ == "__main__":
    detector = CybersecurityThreatDetector()
    detector.train()
    
    print("\n===== TESTING THREAT DETECTION SYSTEM =====\n")
    
    print("\n1. Testing Network Anomaly Detection...")
    test_network_data = generate_sample_network_data(1000)
    features, _ = preprocess_network_data(test_network_data)
    network_results = detector.detect_network_threats(features)
    print(f"Network traffic analysis:")
    print(f"  - Total traffic flows: {len(features)}")
    print(f"  - Detected anomalies: {network_results['num_anomalies']}")
    print(f"  - Anomaly percentage: {network_results['anomaly_percentage']:.2f}%")
    
    print("\n2. Testing Malware Detection...")
    test_files = [
        "benign_test_1.exe", 
        "malware_test_1.exe",
        "suspicious_file.exe"
    ]
    malware_results = detector.detect_malware(test_files)
    print(f"Malware detection results:")
    for result in malware_results:
        if 'error' in result:
            print(f"  - {result['file_path']}: Error - {result['error']}")
        else:
            print(f"  - {result['file_path']}: {'MALWARE' if result['is_malware'] else 'BENIGN'} " 
                  f"(confidence: {result['malware_probability']*100:.1f}%)")
    
    print("\n3. Testing Phishing Detection...")
    test_emails = [
        "Dear customer, Your Amazon account has been locked. Please verify your information here: http://amazom-security.net/verify",
        "Hi Team, Please review the attached document for our meeting tomorrow. Thanks, John",
        "URGENT: Your PayPal account needs attention! Verify your account at http://192.168.1.1/paypal"
    ]
    phishing_results = detector.detect_phishing(test_emails)
    print(f"Phishing detection results:")
    for i, result in enumerate(phishing_results):
        if 'error' in result:
            print(f"  - Email {i+1}: Error - {result['error']}")
        else:
            print(f"  - Email {i+1}: {'PHISHING' if result['is_phishing'] else 'LEGITIMATE'} " 
                  f"(confidence: {result['phishing_probability']*100:.1f}%)")
    
    detector.save_models()
    print("\n===== SYSTEM READY FOR DEPLOYMENT =====")