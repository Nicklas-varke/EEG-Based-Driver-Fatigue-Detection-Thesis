
# -*- coding: utf-8 -*-
"""
ΒΗΜΑ 2: Προεπεξεργασία, Epoching & Labeling
===========================================

Αυτό το script:
1. Φορτώνει τα EEG δεδομένα από το βήμα 1
2. Εφαρμόζει προεπεξεργασία (φίλτρα, downsampling)
3. Μετατρέπει annotations σε events  
4. Δημιουργεί 1-second epochs
5. Εφαρμόζει time-based labeling (Alert vs Fatigue)
6. Αποθηκεύει το processed dataset

Χρήση:
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Εισαγωγή MNE για EEG επεξεργασία
try:
    import mne
    print("✅ MNE library φορτώθηκε επιτυχώς!")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Τρέξε πρώτα: pip install mne")
    exit()

# Ρυθμίσεις
mne.set_log_level('WARNING')

# OUTPUT PATH - ΑΛΛΑΞΕ ΤΟ ΑΝ ΧΡΕΙΑΖΕΤΑΙ
OUTPUT_PATH = r"C:\Users\nikos22594\python_code"

class EEGPreprocessor:
    """
    Κλάση για προεπεξεργασία EEG δεδομένων
    """
    
    def __init__(self, target_sfreq=100, l_freq=1.0, h_freq=50.0):
        """
        Αρχικοποίηση preprocessor
        
        Args:
            target_sfreq: Στόχος συχνότητας (100 Hz όπως στο paper)
            l_freq: Κάτω συχνότητα φίλτρου (1 Hz)  
            h_freq: Άνω συχνότητα φίλτρου (50 Hz)
        """
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        
    def preprocess_raw(self, raw, subject_name):
        """
        Προεπεξεργασία ενός raw EEG αρχείου
        
        Args:
            raw: MNE Raw object
            subject_name: Όνομα subject για logging
            
        Returns:
            processed_raw: Προεπεξεργασμένο Raw object
        """
        print(f"\n🔧 Προεπεξεργασία {subject_name}...")
        
        # Αντίγραφο για να μην αλλάξουμε το πρωτότυπο
        raw_copy = raw.copy()
        
        # 1. Φιλτράρισμα (1-50 Hz όπως στο paper)
        print(f"   📡 Εφαρμογή bandpass filter: {self.l_freq}-{self.h_freq} Hz")
        raw_copy.filter(l_freq=self.l_freq, h_freq=self.h_freq, 
                       fir_design='firwin', verbose=False)
        
        # 2. Resampling στα 100 Hz (όπως στο paper)
        if raw_copy.info['sfreq'] != self.target_sfreq:
            print(f"   📉 Resampling: {raw_copy.info['sfreq']} Hz → {self.target_sfreq} Hz")
            raw_copy.resample(sfreq=self.target_sfreq, verbose=False)
        
        # 3. Εξασφάλιση ότι έχουμε ακριβώς 30 κανάλια
        if len(raw_copy.ch_names) != 30:
            print(f"   ⚠️  Προσοχή: {len(raw_copy.ch_names)} κανάλια αντί για 30")
            
        duration_min = raw_copy.times[-1] / 60
        print(f"   ✅ Προεπεξεργασία ολοκληρώθηκε: {duration_min:.1f} λεπτά, {self.target_sfreq} Hz")
        
        return raw_copy

class EpochGenerator:
    """
    Κλάση για δημιουργία epochs από EEG δεδομένα
    """
    
    def __init__(self, epoch_length=1.0, overlap=0.0):
        """
        Αρχικοποίηση epoch generator
        
        Args:
            epoch_length: Διάρκεια epoch σε δευτερόλεπτα (1.0 όπως στο paper)
            overlap: Επικάλυψη μεταξύ epochs (0.0 = χωρίς επικάλυψη)
        """
        self.epoch_length = epoch_length
        self.overlap = overlap
        
    def create_epochs(self, raw, subject_name):
        """
        Δημιουργία epochs από raw data
        
        Args:
            raw: Προεπεξεργασμένο Raw object
            subject_name: Όνομα subject
            
        Returns:
            epochs_data: numpy array (n_epochs, n_channels, n_timepoints)
            epochs_times: array με χρόνους κάθε epoch
        """
        print(f"\n📦 Δημιουργία epochs για {subject_name}...")
        
        # Παραμέτρους
        sfreq = raw.info['sfreq']
        n_channels = raw.info['nchan']
        samples_per_epoch = int(self.epoch_length * sfreq)
        step_size = int(samples_per_epoch * (1 - self.overlap))
        
        # Δεδομένα EEG
        data = raw.get_data()  # Shape: (n_channels, n_timepoints)
        n_timepoints = data.shape[1]
        
        print(f"   📊 Δεδομένα: {n_channels} κανάλια, {n_timepoints} timepoints")
        print(f"   ⏱️  Epoch: {self.epoch_length} sec = {samples_per_epoch} samples")
        print(f"   👣 Step size: {step_size} samples (overlap: {self.overlap})")
        
        # Δημιουργία epochs
        epochs_list = []
        epoch_times = []
        
        for start_idx in range(0, n_timepoints - samples_per_epoch + 1, step_size):
            end_idx = start_idx + samples_per_epoch
            
            # Εξαγωγή epoch
            epoch = data[:, start_idx:end_idx]
            
            # Έλεγχος μεγέθους
            if epoch.shape[1] == samples_per_epoch:
                epochs_list.append(epoch)
                epoch_start_time = start_idx / sfreq
                epoch_times.append(epoch_start_time)
        
        epochs_data = np.array(epochs_list)
        epoch_times = np.array(epoch_times)
        
        print(f"   ✅ Δημιουργήθηκαν {len(epochs_data)} epochs")
        print(f"   📏 Shape: {epochs_data.shape}")
        
        return epochs_data, epoch_times

class TimeLabelGenerator:
    """
    Κλάση για δημιουργία time-based labels
    """
    
    def __init__(self, alert_duration=30, fatigue_duration=30):
        """
        Αρχικοποίηση label generator
        
        Args:
            alert_duration: Διάρκεια alert period σε λεπτά
            fatigue_duration: Διάρκεια fatigue period σε λεπτά
        """
        self.alert_duration = alert_duration
        self.fatigue_duration = fatigue_duration
        
    def generate_labels(self, epoch_times, total_duration_min, subject_name):
        """
        Δημιουργία time-based labels
        
        Args:
            epoch_times: Array με χρόνους epochs σε δευτερόλεπτα
            total_duration_min: Συνολική διάρκεια σε λεπτά
            subject_name: Όνομα subject
            
        Returns:
            labels: Array με labels (0=Alert, 1=Fatigue)
            label_info: Dictionary με πληροφορίες labeling
        """
        print(f"\n🏷️  Δημιουργία labels για {subject_name}...")
        
        total_duration_sec = total_duration_min * 60
        alert_end_sec = self.alert_duration * 60
        fatigue_start_sec = total_duration_sec - (self.fatigue_duration * 60)
        
        print(f"   📅 Συνολική διάρκεια: {total_duration_min:.1f} λεπτά")
        print(f"   ✅ Alert period: 0 - {self.alert_duration} λεπτά")
        print(f"   😴 Fatigue period: {total_duration_min - self.fatigue_duration:.1f} - {total_duration_min:.1f} λεπτά")
        
        # Δημιουργία labels
        labels = []
        alert_count = 0
        fatigue_count = 0
        excluded_count = 0
        
        for epoch_time in epoch_times:
            if epoch_time <= alert_end_sec:
                labels.append(0)  # Alert
                alert_count += 1
            elif epoch_time >= fatigue_start_sec:
                labels.append(1)  # Fatigue  
                fatigue_count += 1
            else:
                labels.append(-1)  # Transition period (θα αφαιρεθεί)
                excluded_count += 1
        
        labels = np.array(labels)
        
        # Αφαίρεση transition epochs
        valid_indices = labels != -1
        filtered_labels = labels[valid_indices]
        
        print(f"   📊 Alert epochs: {alert_count}")
        print(f"   📊 Fatigue epochs: {fatigue_count}")
        print(f"   📊 Transition epochs (αφαιρούνται): {excluded_count}")
        print(f"   📊 Τελικά epochs: {len(filtered_labels)}")
        
        label_info = {
            'alert_count': alert_count,
            'fatigue_count': fatigue_count,
            'excluded_count': excluded_count,
            'total_valid': len(filtered_labels),
            'valid_indices': valid_indices,
            'alert_duration': self.alert_duration,
            'fatigue_duration': self.fatigue_duration
        }
        
        return filtered_labels, label_info

def find_set_files(base_dir):
    """Βρίσκει όλα τα .set αρχεία (copy από step 1)"""
    set_files = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if item.endswith('.set') and os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                if subitem.endswith('.set'):
                    subfile_path = os.path.join(item_path, subitem)
                    if os.path.isfile(subfile_path):
                        set_files.append(subfile_path)
        elif os.path.isdir(item_path) and item.startswith('s'):
            for subitem in os.listdir(item_path):
                if subitem.endswith('.set'):
                    subfile_path = os.path.join(item_path, subitem)
                    if os.path.isfile(subfile_path):
                        set_files.append(subfile_path)
    
    return sorted(set_files, key=lambda x: os.path.basename(x))

def load_eeg_file(filepath):
    """Φορτώνει ένα EEGLAB αρχείο (copy από step 1)"""
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"❌ Σφάλμα φόρτωσης {filepath}: {e}")
        return None

def main():
    """
    Κύρια συνάρτηση - πλήρης pipeline προεπεξεργασίας
    """
    print("🧠 EEG PREPROCESSING & EPOCHING - ΒΗΜΑ 2")
    print("=" * 60)
    print(f"📅 Ημερομηνία εκτέλεσης: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. Φόρτωση δεδομένων
    print("\n📂 Φάση 1: Φόρτωση EEG δεδομένων")
    print("-" * 40)
    
    current_dir = os.getcwd()
    set_files = find_set_files(current_dir)
    
    if not set_files:
        print("❌ Δεν βρέθηκαν .set αρχεία!")
        return None
    
    print(f"✅ Βρέθηκαν {len(set_files)} subjects")
    
    # Φόρτωση όλων των subjects
    raw_data = {}
    for filepath in set_files:
        subject_name = os.path.basename(filepath).replace('.set', '')
        print(f"   📁 Φορτώνω {subject_name}...")
        
        raw = load_eeg_file(filepath)
        if raw is not None:
            raw_data[subject_name] = raw
        else:
            print(f"   ❌ Αποτυχία φόρτωσης {subject_name}")
    
    print(f"✅ Φορτώθηκαν επιτυχώς: {len(raw_data)} subjects")
    
    # 2. Προεπεξεργασία
    print(f"\n🔧 Φάση 2: Προεπεξεργασία EEG σημάτων")
    print("-" * 40)
    
    preprocessor = EEGPreprocessor(target_sfreq=100, l_freq=1.0, h_freq=50.0)
    processed_data = {}
    
    for subject_name, raw in raw_data.items():
        processed_raw = preprocessor.preprocess_raw(raw, subject_name)
        processed_data[subject_name] = processed_raw
    
    print(f"✅ Προεπεξεργασία ολοκληρώθηκε για {len(processed_data)} subjects")
    
    # 3. Epoching
    print(f"\n📦 Φάση 3: Δημιουργία Epochs")
    print("-" * 40)
    
    epoch_generator = EpochGenerator(epoch_length=1.0, overlap=0.0)
    epoched_data = {}
    
    for subject_name, processed_raw in processed_data.items():
        epochs_data, epoch_times = epoch_generator.create_epochs(processed_raw, subject_name)
        epoched_data[subject_name] = {
            'epochs': epochs_data,
            'times': epoch_times,
            'duration_min': processed_raw.times[-1] / 60
        }
    
    # 4. Labeling
    print(f"\n🏷️  Φάση 4: Time-based Labeling")
    print("-" * 40)
    
    label_generator = TimeLabelGenerator(alert_duration=30, fatigue_duration=30)
    final_dataset = {}
    
    total_epochs = 0
    total_alert = 0
    total_fatigue = 0
    
    for subject_name, epoch_data in epoched_data.items():
        epochs = epoch_data['epochs']
        times = epoch_data['times']
        duration = epoch_data['duration_min']
        
        # Δημιουργία labels
        labels, label_info = label_generator.generate_labels(times, duration, subject_name)
        
        # Φιλτράρισμα epochs με βάση τα valid labels
        valid_indices = label_info['valid_indices']
        filtered_epochs = epochs[valid_indices]
        
        # Αποθήκευση
        final_dataset[subject_name] = {
            'epochs': filtered_epochs,
            'labels': labels,
            'label_info': label_info,
            'original_epochs': len(epochs),
            'valid_epochs': len(filtered_epochs)
        }
        
        total_epochs += len(filtered_epochs)
        total_alert += label_info['alert_count']
        total_fatigue += label_info['fatigue_count']
    
    # 5. Συνολικά στατιστικά
    print(f"\n📊 Φάση 5: Τελικά Στατιστικά Dataset")
    print("=" * 60)
    
    print(f"📈 Συνολικά Αποτελέσματα:")
    print(f"   📦 Subjects: {len(final_dataset)}")
    print(f"   📦 Συνολικά epochs: {total_epochs:,}")
    print(f"   📦 Alert epochs: {total_alert:,} ({total_alert/total_epochs*100:.1f}%)")
    print(f"   📦 Fatigue epochs: {total_fatigue:,} ({total_fatigue/total_epochs*100:.1f}%)")
    print(f"   📦 Μέγεθος epoch: (30, 100) - 30 κανάλια × 100 timepoints")
    
    # Εκτίμηση μεγέθους
    epoch_size_bytes = 30 * 100 * 4  # 4 bytes per float32
    total_size_mb = (total_epochs * epoch_size_bytes) / (1024 * 1024)
    print(f"   💾 Εκτιμώμενο μέγεθος: {total_size_mb:.1f} MB")
    
    print(f"\n📋 Ανά Subject:")
    for subject_name, data in final_dataset.items():
        alert_pct = data['label_info']['alert_count'] / data['valid_epochs'] * 100
        fatigue_pct = data['label_info']['fatigue_count'] / data['valid_epochs'] * 100
        print(f"   {subject_name:15s}: {data['valid_epochs']:4d} epochs "
              f"(Alert: {alert_pct:4.1f}%, Fatigue: {fatigue_pct:4.1f}%)")
    
    # 6. Αποθήκευση dataset
    print(f"\n💾 Φάση 6: Αποθήκευση Processed Dataset")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_eeg_dataset_{timestamp}.pkl"
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    
    # Δημιουργία τελικού dataset dictionary
    save_data = {
        'dataset': final_dataset,
        'metadata': {
            'total_subjects': len(final_dataset),
            'total_epochs': total_epochs,
            'alert_epochs': total_alert,
            'fatigue_epochs': total_fatigue,
            'preprocessing': {
                'target_sfreq': 100,
                'l_freq': 1.0,
                'h_freq': 50.0
            },
            'epoching': {
                'epoch_length': 1.0,
                'overlap': 0.0
            },
            'labeling': {
                'alert_duration': 30,
                'fatigue_duration': 30,
                'method': 'time-based'
            },
            'created': datetime.now().isoformat(),
            'shape_info': {
                'epoch_shape': '(30, 100)',
                'n_channels': 30,
                'n_timepoints': 100,
                'sampling_rate': 100
            }
        }
    }
    
    try:
        with open(output_filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✅ Dataset αποθηκεύτηκε επιτυχώς!")
        print(f"📁 Αρχείο: {output_filename}")
        print(f" Τοποθεσία: {OUTPUT_PATH}")
        print(f" Μέγεθος αρχείου: {os.path.getsize(output_filepath) / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"❌ Σφάλμα αποθήκευσης: {e}")
        return None
    
    print(f"\n🎉 ΟΛΟΚΛΗΡΩΣΗ ΕΠΙΤΥΧΟΥΣ!")
    print("=" * 60)
    print("✅ Το processed dataset είναι έτοιμο για machine learning!")
    print("📋 Επόμενα βήματα:")
    print("   1️⃣  Feature extraction")
    print("   2️⃣  ESTCNN model implementation")
    print("   3️⃣  Baseline models (SVM, LSTM, etc.)")
    print("   4️⃣  Cross-validation evaluation")
    
    return save_data

# Εκτέλεση με αυτόματη αποθήκευση report
if __name__ == "__main__":
    # Δημιουργία output report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"EEG_Preprocessing_Report_{timestamp}.txt"
    report_filepath = os.path.join(OUTPUT_PATH, report_filename)
    
    print(f"🚀 Ξεκινάει EEG Preprocessing Pipeline...")
    print(f"📄 Report θα αποθηκευτεί: {report_filename}")
    print("=" * 60)
    
    # Εκτέλεση main με output capture
    try:
        # Simple output capture
        original_stdout = sys.stdout
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            # Redirect stdout to both console and file
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
            
            # Εκτέλεση main
            result = main()
            
            # Restore stdout
            sys.stdout = original_stdout
            
        print(f"\n📄 Report αποθηκεύτηκε: {report_filename}")
        
        if result is not None:
            print("🎉 Preprocessing ολοκληρώθηκε επιτυχώς!")
        else:
            print("❌ Κάτι πήγε στραβά στο preprocessing.")
            
    except Exception as e:
        sys.stdout = original_stdout
        print(f"❌ Σφάλμα: {e}")




