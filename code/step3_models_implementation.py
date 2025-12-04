# -*- coding: utf-8 -*-
"""
ΒΗΜΑ 3: ESTCNN & Baseline Models Implementation
===============================================

Αυτό το script:
1. Φορτώνει το processed dataset από το βήμα 2
2. Υλοποιεί το ESTCNN model (κυρίως μοντέλο)
3. Υλοποιεί baseline models (PSD-SVM, LSTM, CNN variants)
4. Εξάγει features για traditional ML methods
5. Ετοιμάζει training pipeline με cross-validation

Χρήση:
Τρέξε αυτό το script μετά το step2_preprocessing_epoching.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & ML
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    print(f"✅ TensorFlow {tf.__version__} φορτώθηκε επιτυχώς!")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Τρέξε πρώτα: pip install tensorflow")
    exit()

try:
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import pandas as pd
    print("✅ Scikit-learn & pandas φορτώθηκαν επιτυχώς!")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Τρέξε πρώτα: pip install scikit-learn pandas")
    exit()

try:
    from scipy import signal
    from scipy.stats import skew, kurtosis
    print("✅ SciPy φορτώθηκε επιτυχώς!")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Τρέξε πρώτα: pip install scipy")
    exit()

# OUTPUT PATH
OUTPUT_PATH = r"C:\Users\nikos22594\python_code"

class ESTCNNModel:
    """
    EEG-based Spatio-Temporal Convolutional Neural Network (ESTCNN)
    
    Βασισμένο στο paper: "EEG-Based Spatio–Temporal Convolutional Neural 
    Network for Driver Fatigue Evaluation" by Gao et al.
    """
    
    def __init__(self, input_shape=(30, 100, 1), num_classes=2):
        """
        Αρχικοποίηση ESTCNN
        
        Args:
            input_shape: (channels, timepoints, features) = (30, 100, 1)
            num_classes: 2 (Alert vs Fatigue)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def create_core_block(self, inputs, filters, pool_size, pool_type='max', block_name="core"):
        """
        Δημιουργία Core Block σύμφωνα με το paper
        
        Core Block = 3×(Conv1D + ReLU + BatchNorm) + Pooling
        """
        x = inputs
        
        # 3 Convolutional layers με kernel size 3
        for i in range(3):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding='valid',
                activation='relu',
                name=f'{block_name}_conv_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'{block_name}_bn_{i+1}')(x)
        
        # Pooling layer
        if pool_type == 'max':
            x = layers.MaxPooling1D(
                pool_size=pool_size,
                name=f'{block_name}_maxpool'
            )(x)
        else:  # average pooling
            x = layers.AveragePooling1D(
                pool_size=pool_size,
                name=f'{block_name}_avgpool'
            )(x)
        
        return x
    
    def build_model(self):
        """
        Κατασκευή του ESTCNN μοντέλου σύμφωνα με το paper
        
        Architecture:
        - Core Block 1: filters=16, max pooling (pool_size=2)
        - Core Block 2: filters=32, max pooling (pool_size=2)  
        - Core Block 3: filters=64, avg pooling (pool_size=7)
        - Dense Layer: 50 neurons
        - Output Layer: 2 neurons (softmax)
        """
        print("🧠 Κατασκευάζω ESTCNN model...")
        
        # Input layer: (None, 30, 100, 1) για CNN2D ή (None, 100, 30) για CNN1D
        # Χρησιμοποιούμε CNN1D για temporal convolutions
        inputs = keras.Input(shape=(100, 30), name='eeg_input')  # (timepoints, channels)
        
        print(f"   📊 Input shape: {inputs.shape}")
        
        # Core Block 1: 16 filters, max pooling size 2
        x = self.create_core_block(inputs, filters=16, pool_size=2, 
                                 pool_type='max', block_name='core1')
        print(f"   🔧 After Core Block 1: temporal dim reduced by ~4x")
        
        # Core Block 2: 32 filters, max pooling size 2
        x = self.create_core_block(x, filters=32, pool_size=2, 
                                 pool_type='max', block_name='core2')
        print(f"   🔧 After Core Block 2: temporal dim reduced further")
        
        # Core Block 3: 64 filters, average pooling size 7
        x = self.create_core_block(x, filters=64, pool_size=7, 
                                 pool_type='avg', block_name='core3')
        print(f"   🔧 After Core Block 3: temporal features extracted")
        
        # Flatten για dense layers (spatial feature fusion)
        x = layers.Flatten(name='flatten')(x)
        
        # Dense layer για spatial feature fusion
        x = layers.Dense(50, activation='relu', name='dense_spatial')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Δημιουργία μοντέλου
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='ESTCNN')
        
        print(f"   ✅ ESTCNN model δημιουργήθηκε επιτυχώς!")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile του μοντέλου με optimizer και loss function
        """
        if self.model is None:
            self.build_model()
        
        # SGD optimizer όπως στο paper
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"   ✅ ESTCNN compiled με SGD optimizer (lr={learning_rate})")
    
    def summary(self):
        """
        Εμφάνιση summary του μοντέλου
        """
        if self.model is None:
            self.build_model()
        
        print("\n📋 ESTCNN Model Architecture:")
        print("=" * 50)
        self.model.summary()
        return self.model


class BaselineModels:
    """
    Baseline μοντέλα για σύγκριση με το ESTCNN
    """
    
    @staticmethod
    def create_simple_cnn():
        """
        Απλό CNN baseline
        """
        model = models.Sequential([
            layers.Conv1D(32, 7, activation='relu', input_shape=(100, 30)),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 5, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ], name='Simple_CNN')
        
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    @staticmethod
    def create_lstm_model():
        """
        LSTM model για temporal dependencies
        """
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(100, 30)),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ], name='LSTM_Model')
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    @staticmethod
    def create_cnn_lstm_hybrid():
        """
        CNN-LSTM hybrid model
        """
        model = models.Sequential([
            layers.Conv1D(32, 5, activation='relu', input_shape=(100, 30)),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.LSTM(32, return_sequences=False),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ], name='CNN_LSTM_Hybrid')
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model


class FeatureExtractor:
    """
    Εξαγωγή features για traditional ML methods
    """
    
    def __init__(self, sfreq=100):
        self.sfreq = sfreq
        
    def extract_statistical_features(self, epochs):
        """
        Εξαγωγή στατιστικών χαρακτηριστικών
        
        Args:
            epochs: (n_epochs, n_channels, n_timepoints)
            
        Returns:
            features: (n_epochs, n_features)
        """
        print("   📊 Εξαγωγή στατιστικών features...")
        
        n_epochs, n_channels, n_timepoints = epochs.shape
        features_list = []
        
        for epoch in epochs:
            epoch_features = []
            
            for ch in range(n_channels):
                signal_ch = epoch[ch, :]
                
                # Βασικά στατιστικά
                mean_val = np.mean(signal_ch)
                std_val = np.std(signal_ch)
                var_val = np.var(signal_ch)
                
                # Στατιστικά ανώτερης τάξης
                skew_val = skew(signal_ch)
                kurt_val = kurtosis(signal_ch)
                
                # Min/Max values
                min_val = np.min(signal_ch)
                max_val = np.max(signal_ch)
                
                # Range
                range_val = max_val - min_val
                
                epoch_features.extend([mean_val, std_val, var_val, skew_val, 
                                     kurt_val, min_val, max_val, range_val])
            
            features_list.append(epoch_features)
        
        features = np.array(features_list)
        print(f"      ✅ Statistical features shape: {features.shape}")
        return features
    
    def extract_spectral_features(self, epochs):
        """
        Εξαγωγή φασματικών χαρακτηριστικών (PSD, band powers)
        """
        print("   📊 Εξαγωγή spectral features...")
        
        n_epochs, n_channels, n_timepoints = epochs.shape
        features_list = []
        
        # Ορισμός frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for epoch in epochs:
            epoch_features = []
            
            for ch in range(n_channels):
                signal_ch = epoch[ch, :]
                
                # Power Spectral Density
                freqs, psd = signal.welch(signal_ch, fs=self.sfreq, 
                                        nperseg=min(64, len(signal_ch)))
                
                # Band powers
                for band_name, (low_freq, high_freq) in bands.items():
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.mean(psd[band_mask])
                    epoch_features.append(band_power)
                
                # Spectral statistics
                dominant_freq = freqs[np.argmax(psd)]
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                spectral_rolloff = freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
                
                epoch_features.extend([dominant_freq, spectral_centroid, spectral_rolloff])
            
            features_list.append(epoch_features)
        
        features = np.array(features_list)
        print(f"      ✅ Spectral features shape: {features.shape}")
        return features
    
    def extract_all_features(self, epochs):
        """
        Εξαγωγή όλων των features
        """
        print("📊 Εξαγωγή παραδοσιακών features για ML...")
        
        # Στατιστικά features
        stat_features = self.extract_statistical_features(epochs)
        
        # Φασματικά features
        spectral_features = self.extract_spectral_features(epochs)
        
        # Συνδυασμός όλων των features
        all_features = np.concatenate([stat_features, spectral_features], axis=1)
        
        print(f"✅ Συνολικά features: {all_features.shape}")
        return all_features


def load_processed_dataset(dataset_path=None):
    """
    Φόρτωση του processed dataset από το βήμα 2
    """
    print("📂 Φόρτωση processed dataset...")
    
    if dataset_path is None:
        # Αναζήτηση του πιο πρόσφατου dataset
        files = [f for f in os.listdir(OUTPUT_PATH) if f.startswith('processed_eeg_dataset_')]
        if not files:
            print("❌ Δεν βρέθηκε processed dataset!")
            return None
        
        # Πάρε το πιο πρόσφατο
        latest_file = sorted(files)[-1]
        dataset_path = os.path.join(OUTPUT_PATH, latest_file)
    
    print(f"   📁 Φορτώνω: {os.path.basename(dataset_path)}")
    
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   ✅ Dataset φορτώθηκε επιτυχώς!")
        return data
    
    except Exception as e:
        print(f"   ❌ Σφάλμα φόρτωσης: {e}")
        return None


def prepare_data_for_models(dataset_dict):
    """
    Προετοιμασία δεδομένων για τα μοντέλα
    """
    print("🔧 Προετοιμασία δεδομένων για models...")
    
    # Εξαγωγή epochs και labels από όλα τα subjects
    all_epochs = []
    all_labels = []
    subject_info = []
    
    for subject_name, subject_data in dataset_dict['dataset'].items():
        epochs = subject_data['epochs']
        labels = subject_data['labels']
        
        all_epochs.append(epochs)
        all_labels.append(labels)
        
        # Πληροφορίες για cross-validation
        subject_indices = [subject_name] * len(epochs)
        subject_info.extend(subject_indices)
        
        print(f"   📊 {subject_name}: {len(epochs)} epochs")
    
    # Συνδυασμός όλων των δεδομένων
    X = np.concatenate(all_epochs, axis=0)  # (total_epochs, 30, 100)
    y = np.concatenate(all_labels, axis=0)  # (total_epochs,)
    subjects = np.array(subject_info)       # (total_epochs,) με subject names
    
    print(f"✅ Συνολικά δεδομένα:")
    print(f"   📦 X shape: {X.shape}")
    print(f"   🏷️  y shape: {y.shape}")
    print(f"   👥 Subjects: {len(np.unique(subjects))}")
    print(f"   📊 Class distribution: {np.bincount(y)}")
    
    return X, y, subjects


def main():
    """
    Κύρια συνάρτηση - υλοποίηση όλων των μοντέλων
    """
    print("🧠 MODEL IMPLEMENTATION & FEATURE EXTRACTION - ΒΗΜΑ 3")
    print("=" * 70)
    print(f"📅 Ημερομηνία εκτέλεσης: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 1. Φόρτωση processed dataset
    print(f"\n📂 Φάση 1: Φόρτωση Dataset")
    print("-" * 40)
    
    dataset = load_processed_dataset()
    if dataset is None:
        print("❌ Αδυναμία φόρτωσης dataset!")
        return None
    
    # Εμφάνιση metadata
    metadata = dataset['metadata']
    print(f"   📊 Total subjects: {metadata['total_subjects']}")
    print(f"   📊 Total epochs: {metadata['total_epochs']:,}")
    print(f"   📊 Alert epochs: {metadata['alert_epochs']:,}")
    print(f"   📊 Fatigue epochs: {metadata['fatigue_epochs']:,}")
    
    # 2. Προετοιμασία δεδομένων
    print(f"\n🔧 Φάση 2: Προετοιμασία Δεδομένων")
    print("-" * 40)
    
    X, y, subjects = prepare_data_for_models(dataset)
    
    # 3. Feature extraction για traditional ML
    print(f"\n📊 Φάση 3: Feature Extraction")
    print("-" * 40)
    
    feature_extractor = FeatureExtractor(sfreq=100)
    traditional_features = feature_extractor.extract_all_features(X)
    
    # 4. ESTCNN Model Implementation
    print(f"\n🧠 Φάση 4: ESTCNN Model Implementation")
    print("-" * 50)
    
    estcnn = ESTCNNModel(input_shape=(100, 30), num_classes=2)
    estcnn_model = estcnn.build_model()
    estcnn.compile_model(learning_rate=0.001)
    estcnn.summary()
    
    # 5. Baseline Models Implementation
    print(f"\n🏗️  Φάση 5: Baseline Models Implementation")
    print("-" * 50)
    
    print("   🔧 Δημιουργώντας baseline models...")
    
    # Deep Learning baselines
    simple_cnn = BaselineModels.create_simple_cnn()
    lstm_model = BaselineModels.create_lstm_model()
    cnn_lstm_hybrid = BaselineModels.create_cnn_lstm_hybrid()
    
    print("   ✅ Simple CNN model")
    print("   ✅ LSTM model") 
    print("   ✅ CNN-LSTM hybrid model")
    
    # Traditional ML models
    print("   🔧 Δημιουργώντας traditional ML models...")
    
    svm_model = SVC(kernel='rbf', random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("   ✅ SVM model")
    print("   ✅ Random Forest model")
    
    # 6. Αποθήκευση μοντέλων και δεδομένων
    print(f"\n💾 Φάση 6: Αποθήκευση Models & Data")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Αποθήκευση δεδομένων για training
    ml_data = {
        'X_raw': X,  # Raw epochs για deep learning
        'X_features': traditional_features,  # Extracted features για ML
        'y': y,
        'subjects': subjects,
        'metadata': {
            'n_samples': len(X),
            'n_channels': X.shape[1],
            'n_timepoints': X.shape[2],
            'n_features': traditional_features.shape[1],
            'n_classes': len(np.unique(y)),
            'class_names': ['Alert', 'Fatigue'],
            'sampling_rate': 100,
            'created': datetime.now().isoformat()
        }
    }
    
    # Αποθήκευση δεδομένων
    data_filename = f"ml_ready_data_{timestamp}.pkl"
    data_filepath = os.path.join(OUTPUT_PATH, data_filename)
    
    with open(data_filepath, 'wb') as f:
        pickle.dump(ml_data, f)
    
    print(f"✅ ML data αποθηκεύτηκαν: {data_filename}")
    
    # Αποθήκευση ESTCNN model
    estcnn_filename = f"estcnn_model_{timestamp}.h5"
    estcnn_filepath = os.path.join(OUTPUT_PATH, estcnn_filename)
    estcnn_model.save(estcnn_filepath)
    
    print(f"✅ ESTCNN model αποθηκεύτηκε: {estcnn_filename}")
    
    # 7. Τελικά στατιστικά
    print(f"\n📈 Φάση 7: Τελικά Στατιστικά")
    print("=" * 50)
    
    print(f"🎯 Έτοιμα Models:")
    print(f"   ✅ ESTCNN (κύριο μοντέλο)")
    print(f"   ✅ Simple CNN (baseline)")
    print(f"   ✅ LSTM (baseline)")
    print(f"   ✅ CNN-LSTM Hybrid (baseline)")
    print(f"   ✅ SVM (traditional ML)")
    print(f"   ✅ Random Forest (traditional ML)")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   📦 Raw data shape: {X.shape}")
    print(f"   📦 Features shape: {traditional_features.shape}")
    print(f"   👥 Subjects: {len(np.unique(subjects))}")
    print(f"   🏷️  Classes: {len(np.unique(y))} (Alert: {np.sum(y==0)}, Fatigue: {np.sum(y==1)})")
    
    print(f"\n💾 Αποθηκευμένα Αρχεία:")
    print(f"   📁 {data_filename} ({os.path.getsize(data_filepath)/(1024*1024):.1f} MB)")
    print(f"   📁 {estcnn_filename} ({os.path.getsize(estcnn_filepath)/(1024*1024):.1f} MB)")
    
    print(f"\n🎉 ΟΛΟΚΛΗΡΩΣΗ ΕΠΙΤΥΧΟΥΣ!")
    print("=" * 50)
    print("✅ Όλα τα models είναι έτοιμα για training!")
    print("📋 Επόμενο βήμα: Cross-validation & evaluation")
    
    return {
        'models': {
            'estcnn': estcnn_model,
            'simple_cnn': simple_cnn,
            'lstm': lstm_model,
            'cnn_lstm': cnn_lstm_hybrid,
            'svm': svm_model,
            'random_forest': rf_model
        },
        'data': ml_data,
        'filepaths': {
            'data': data_filepath,
            'estcnn': estcnn_filepath
        }
    }


# Εκτέλεση με report generation
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"Model_Implementation_Report_{timestamp}.txt"
    report_filepath = os.path.join(OUTPUT_PATH, report_filename)
    
    print(f"🚀 Ξεκινάει Model Implementation Pipeline...")
    print(f"📄 Report θα αποθηκευτεί: {report_filename}")
    print("=" * 70)
    
    # Output capture
    try:
        original_stdout = sys.stdout
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            class Tee:
                def __init__(self, console, file):
                    self.console = console
                    self.file = file
                def write(self, message):
                    self.console.write(message)
                    self.file.write(message)
                def flush(self):
                    self.console.flush()
                    self.file.flush()
            
            sys.stdout = Tee(original_stdout, f)
            result = main()
            sys.stdout = original_stdout
            
        print(f"\n📄 Report αποθηκεύτηκε: {report_filename}")
        
        if result is not None:
            print("🎉 Model implementation ολοκληρώθηκε επιτυχώς!")
            print("🚀 Έτοιμο για training και evaluation!")
        else:
            print("❌ Κάτι πήγε στραβά στη model implementation.")
            
    except Exception as e:
        sys.stdout = original_stdout
        print(f"❌ Σφάλμα: {e}")
