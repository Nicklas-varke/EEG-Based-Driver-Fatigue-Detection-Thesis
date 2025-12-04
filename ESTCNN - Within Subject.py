# -*- coding: utf-8 -*-
"""
ESTCNN - Within-Subject Evaluation
=================================================
Κάθε subject εκπαιδεύεται και αξιολογείται ξεχωριστά
"""

import os
import numpy as np
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_PATH = os.getcwd()


def load_ml_ready_data():
    """Φόρτωση data με subject information"""
    print("📂 Φόρτωση ML-ready dataset...")
    
    files = [f for f in os.listdir(OUTPUT_PATH) if f.startswith('ml_ready_data_')]
    if not files:
        print("❌ Δεν βρέθηκε dataset!")
        return None
    
    latest_file = sorted(files)[-1]
    data_path = os.path.join(OUTPUT_PATH, latest_file)
    
    print(f"   📁 {latest_file}")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   ✅ Dataset loaded!")
    return data


def build_estcnn_optimized():
    """
    ΒΕΛΤΙΩΜΕΝΟ ESTCNN (paper with Adam)
    """
    inputs = keras.Input(shape=(100, 30), name='eeg_input')
    
    # Core Block 1: 16 filters
    x = inputs
    for i in range(3):
        x = layers.Conv1D(16, 3, padding='valid', activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Core Block 2: 32 filters
    for i in range(3):
        x = layers.Conv1D(32, 3, padding='valid', activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Core Block 3: 64 filters
    for i in range(3):
        x = layers.Conv1D(64, 3, padding='valid', activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.AveragePooling1D(7)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(50, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='ESTCNN')
    
    # Adam optimizer (καλύτερο από SGD)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def within_subject_cv(X_subject, y_subject, subject_name, n_folds=10):
    """
    Within-subject 10-fold CV για έναν subject
    
    Όπως στο paper: train/test από τον ΙΔΙΟ subject
    """
    print(f"\n{'='*60}")
    print(f" Subject: {subject_name}")
    print(f"{'='*60}")
    print(f"   Total samples: {len(X_subject):,}")
    print(f"   Class distribution: {np.bincount(y_subject)}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_subject, y_subject)):
        print(f"\n   📊 Fold {fold_idx + 1}/{n_folds}")
        
        # Split
        X_train, X_val = X_subject[train_idx], X_subject[val_idx]
        y_train, y_val = y_subject[train_idx], y_subject[val_idx]
        
        print(f"      Train: {len(X_train):,} | Val: {len(X_val):,}")
        
        # Build fresh model
        model = build_estcnn_optimized()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-7,
            verbose=0
        )
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluation
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        
        fold_results.append({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'epochs_trained': len(history.history['loss'])
        })
        
        print(f"      ✅ Acc: {acc:.4f} | Epochs: {len(history.history['loss'])}")
    
    # Subject summary
    mean_acc = np.mean([r['accuracy'] for r in fold_results])
    std_acc = np.std([r['accuracy'] for r in fold_results])
    
    print(f"\n   🎯 {subject_name} Results:")
    print(f"      Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return fold_results, mean_acc, std_acc


def main():
    """
    Within-Subject Evaluation όπως το paper
    """
    print("🧠 ESTCNN - WITHIN-SUBJECT EVALUATION (PAPER METHOD)")
    print("="*70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 1. Load data
    print("\n📂 Βήμα 1: Φόρτωση δεδομένων")
    print("-"*40)
    
    data = load_ml_ready_data()
    if data is None:
        return
    
    X_raw = data['X_raw']
    y = data['y']
    subjects = data['subjects']
    
    print(f"   📊 Total samples: {len(X_raw):,}")
    print(f"   📊 Unique subjects: {len(np.unique(subjects))}")
    
    # Transpose
    print("\n🔧 Βήμα 2: Προετοιμασία δεδομένων")
    print("-"*40)
    
    X = np.transpose(X_raw, (0, 2, 1))
    print(f"   ✅ Shape: {X.shape} (samples, timepoints, channels)")
    
    # 2. Within-subject CV για κάθε subject
    print("\n🎯 Βήμα 3: Within-Subject 10-Fold CV")
    print("-"*40)
    
    all_subject_results = {}
    subject_accuracies = []
    
    unique_subjects = np.unique(subjects)
    
    for subject_name in unique_subjects:
        # Get data για αυτό το subject μόνο
        subject_mask = subjects == subject_name
        X_subject = X[subject_mask]
        y_subject = y[subject_mask]
        
        # Within-subject CV
        fold_results, mean_acc, std_acc = within_subject_cv(
            X_subject, y_subject, subject_name, n_folds=10
        )
        
        all_subject_results[subject_name] = {
            'fold_results': fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }
        
        subject_accuracies.append(mean_acc)
    
    # 3. Overall results
    print("\n" + "="*70)
    print("📈 ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ - WITHIN-SUBJECT")
    print("="*70)
    
    print("\n📊 Per-Subject Accuracies:")
    for subject_name in unique_subjects:
        result = all_subject_results[subject_name]
        print(f"   {subject_name}: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f}")
    
    overall_mean = np.mean(subject_accuracies)
    overall_std = np.std(subject_accuracies)
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Mean Accuracy: {overall_mean:.4f} ± {overall_std:.4f}")
    print(f"   Range: {min(subject_accuracies):.4f} - {max(subject_accuracies):.4f}")
    
    # Comparison with paper
    paper_acc = 0.9737
    print(f"\n📊 Σύγκριση με Paper:")
    print(f"   Paper (SGD, within-subject): {paper_acc:.4f}")
    print(f"   Εμείς (Adam, within-subject): {overall_mean:.4f}")
    
    if overall_mean >= 0.90:
        print(f"\n✅ ΕΞΑΙΡΕΤΙΚΗ ΑΠΌΔΟΣΗ! (≥90%)")
    elif overall_mean >= 0.85:
        print(f"\n✅ ΠΟΛΥ ΚΑΛΗ ΑΠΌΔΟΣΗ! (≥85%)")
    else:
        print(f"\n⚠️  ΜΕΤΡΙΑ ΑΠΌΔΟΣΗ (<85%)")
    
    # 4. Save
    print("\n💾 Βήμα 4: Αποθήκευση")
    print("-"*40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dict = {
        'evaluation_type': 'within_subject',
        'n_folds': 10,
        'subject_results': all_subject_results,
        'overall_mean': overall_mean,
        'overall_std': overall_std,
        'timestamp': timestamp
    }
    
    filename = f"ESTCNN_WithinSubject_Results_{timestamp}.pkl"
    filepath = os.path.join(OUTPUT_PATH, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results_dict, f)
    
    print(f"   ✅ Αποθηκεύτηκε: {filename}")
    
    print("\n" + "="*70)
    print("✅ ΟΛΟΚΛΗΡΩΣΗ!")
    print("="*70)


if __name__ == "__main__":
    main()


