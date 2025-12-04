
# -*- coding: utf-8 -*-
"""
AUTO-EXPORT VERSION: EEG Data Loader με Αυτόματη Αποθήκευση
===========================================================

Αυτό το script:
1. Φορτώνει όλα τα EEG δεδομένα
2. Αποθηκεύει αυτόματα όλο το output σε .txt αρχείο
3. Δημιουργεί formatted report 

Χρήση: Απλά τρέξε το script και θα δημιουργηθεί αυτόματα το report!
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from contextlib import redirect_stdout
import io

# Εισαγωγή MNE για EEGLAB αρχεία
try:
    import mne
    print("✅ MNE library φορτώθηκε επιτυχώς!")
except ImportError:
    print("❌ ΣΦΑΛΜΑ: Τρέξε πρώτα: pip install mne")
    exit()

# Ρυθμίσεις
mne.set_log_level('WARNING')  # Λιγότερα μηνύματα

# OUTPUT PATH - ΑΛΛΑΞΕ ΤΟ ΑΝ ΧΡΕΙΑΖΕΤΑΙ
OUTPUT_PATH = r"C:\Users\nikos22594\python_code"

class OutputCapture:
    """Κλάση για καταγραφή όλου του output"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        
    def __enter__(self):
        # Δημιουργία tee που γράφει και στο console και στο buffer
        sys.stdout = TeeOutput(self.original_stdout, self.buffer)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        
        # Αποθήκευση σε αρχείο
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(self.buffer.getvalue())
        
        print(f"\n📄 Report αποθηκεύτηκε: {self.filepath}")

class TeeOutput:
    """Κλάση που γράφει output και στο console και στο buffer"""
    
    def __init__(self, console, buffer):
        self.console = console
        self.buffer = buffer
        
    def write(self, message):
        self.console.write(message)
        self.buffer.write(message)
        
    def flush(self):
        self.console.flush()
        self.buffer.flush()

def find_set_files(base_dir):
    """
    Βρίσκει όλα τα πραγματικά .set αρχεία μέσα σε φακέλους
    """
    set_files = []
    
    print("🔍 Ψάχνω για .set αρχεία...")
    
    # Ψάξε στον τρέχοντα φάκελο
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Αν είναι αρχείο .set
        if item.endswith('.set') and os.path.isfile(item_path):
            set_files.append(item_path)
            print(f"   ✅ Βρέθηκε αρχείο: {item}")
            
        # Αν είναι φάκελος που τελειώνει σε .set
        elif item.endswith('.set') and os.path.isdir(item_path):
            print(f"   📁 Ψάχνω στον φάκελο: {item}")
            # Ψάξε μέσα στον φάκελο για .set αρχείο
            try:
                for subitem in os.listdir(item_path):
                    if subitem.endswith('.set'):
                        subfile_path = os.path.join(item_path, subitem)
                        if os.path.isfile(subfile_path):
                            set_files.append(subfile_path)
                            print(f"      ✅ Βρέθηκε: {subitem}")
            except Exception as e:
                print(f"      ❌ Σφάλμα ανάγνωσης φακέλου: {e}")
                        
        # Αν είναι φάκελος με όνομα subject (π.χ. s01_051017m)
        elif os.path.isdir(item_path) and item.startswith('s'):
            print(f"   📁 Ψάχνω στον subject φάκελο: {item}")
            # Ψάξε μέσα για .set αρχεία
            try:
                for subitem in os.listdir(item_path):
                    if subitem.endswith('.set'):
                        subfile_path = os.path.join(item_path, subitem)
                        if os.path.isfile(subfile_path):
                            set_files.append(subfile_path)
                            print(f"      ✅ Βρέθηκε: {subitem}")
            except Exception as e:
                print(f"      ❌ Σφάλμα ανάγνωσης φακέλου: {e}")
    
    return set_files

def load_eeg_file(filepath):
    """
    Φορτώνει ένα EEGLAB .set αρχείο
    
    Input: διαδρομή αρχείου .set
    Output: MNE Raw object
    """
    try:
        print(f"\n Φορτώνω: {filepath}")
        
        # Έλεγχος ότι το αρχείο υπάρχει
        if not os.path.isfile(filepath):
            print(f"❌ Το αρχείο δεν υπάρχει: {filepath}")
            return None
            
        # Έλεγχος για το αντίστοιχο .fdt αρχείο (στον ίδιο φάκελο)
        set_dir = os.path.dirname(filepath)
        set_name = os.path.basename(filepath).replace('.set', '')
        fdt_path = os.path.join(set_dir, f"{set_name}.fdt")
        
        if not os.path.isfile(fdt_path):
            print(f"⚠️  Προειδοποίηση: Δεν βρέθηκε το .fdt αρχείο: {fdt_path}")
        else:
            print(f"✅ Βρέθηκε και το .fdt αρχείο!")
        
        # Φόρτωση EEGLAB αρχείου
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        
        print(f"✅ Επιτυχής φόρτωση!")
        return raw
        
    except Exception as e:
        print(f"❌ Σφάλμα φόρτωσης: {e}")
        return None

def show_eeg_info(raw, subject_name):
    """
    Δείχνει βασικές πληροφορίες για το EEG αρχείο
    """
    if raw is None:
        return False
        
    print(f"\n📊 ΠΛΗΡΟΦΟΡΙΕΣ ΓΙΑ {subject_name}")
    print("=" * 50)
    
    # Βασικές πληροφορίες
    n_channels = raw.info['nchan']
    sfreq = raw.info['sfreq']
    duration = raw.times[-1]
    
    print(f"🧠 Κανάλια EEG: {n_channels}")
    print(f" Συχνότητα δειγματοληψίας: {sfreq} Hz")
    print(f"⏱️  Διάρκεια: {duration:.1f} δευτερόλεπτα ({duration/60:.1f} λεπτά)")
    print(f"📏 Συνολικά samples: {len(raw.times)}")
    
    # Ονόματα καναλιών
    print(f"\n🔧 Ονόματα καναλιών (πρώτα 10):")
    channel_names = raw.ch_names[:10]
    print(", ".join(channel_names))
    if len(raw.ch_names) > 10:
        print(f"... και άλλα {len(raw.ch_names) - 10}")
    
    # Events - δοκιμάζω διαφορετικές μεθόδους
    print(f"\n🎯 Αναζήτηση Events...")
    
    events_found = False
    
    # Μέθοδος 1: Κανονική αναζήτηση events
    try:
        events = mne.find_events(raw, verbose=False)
        if len(events) > 0:
            print(f"✅ Events βρέθηκαν (μέθοδος 1): {len(events)}")
            
            # Τύποι events
            event_types = np.unique(events[:, 2])
            print(f"📋 Τύποι events: {event_types}")
            
            # Μετράμε κάθε τύπο
            for event_type in event_types:
                count = np.sum(events[:, 2] == event_type)
                print(f"   Event {event_type}: {count} φορές")
            
            events_found = True
            
    except Exception as e:
        print(f"⚠️  Μέθοδος 1 απέτυχε: {e}")
    
    # Μέθοδος 2: Αναζήτηση με διαφορετικές παραμέτρους
    if not events_found:
        try:
            events = mne.find_events(raw, stim_channel='auto', verbose=False)
            if len(events) > 0:
                print(f"✅ Events βρέθηκαν (μέθοδος 2): {len(events)}")
                event_types = np.unique(events[:, 2])
                print(f"📋 Τύποι events: {event_types}")
                events_found = True
        except Exception as e:
            print(f"⚠️  Μέθοδος 2 απέτυχε: {e}")
    
    # Μέθοδος 3: Ελέγχω αν υπάρχει STI κανάλι
    if not events_found:
        try:
            stim_channels = [ch for ch in raw.ch_names if 'STI' in ch.upper() or 'TRIG' in ch.upper()]
            if stim_channels:
                print(f"✅ Βρέθηκαν stimulus κανάλια: {stim_channels}")
                for stim_ch in stim_channels:
                    events = mne.find_events(raw, stim_channel=stim_ch, verbose=False)
                    if len(events) > 0:
                        print(f"✅ Events από {stim_ch}: {len(events)}")
                        event_types = np.unique(events[:, 2])
                        print(f"📋 Τύποι events: {event_types}")
                        events_found = True
                        break
            else:
                print("❌ Δεν βρέθηκαν stimulus κανάλια")
        except Exception as e:
            print(f"⚠️  Μέθοδος 3 απέτυχε: {e}")
    
    # Αν δεν βρέθηκαν events
    if not events_found:
        print("❌ Δεν βρέθηκαν events με καμία μέθοδο")
        print("💡 Αυτό μπορεί να σημαίνει:")
        print("   - Τα events είναι σε διαφορετικό format")
        print("   - Χρειάζεται διαφορετική προεπεξεργασία")
        print("   - Τα events είναι embedded στο αρχείο με άλλο τρόπο")
        
        # Εμφάνιση όλων των καναλιών για debugging (μόνο τα πρώτα 15 για συντομία)
        print(f"\n🔍 Κανάλια EEG (πρώτα 15 από {len(raw.ch_names)}):")
        for i, ch_name in enumerate(raw.ch_names[:15]):
            ch_type = raw.get_channel_types()[i]
            print(f"   {i+1:2d}. {ch_name} ({ch_type})")
        if len(raw.ch_names) > 15:
            print(f"   ... και άλλα {len(raw.ch_names) - 15} κανάλια")
    
    return events_found

def main():
    """
    Κύρια συνάρτηση - φορτώνει όλα τα subjects
    """
    print("🧠 EEG DATA LOADER - REPORT GENERATOR")
    print("=" * 60)
    print(f"📅 Ημερομηνία εκτέλεσης: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Working Directory: {os.getcwd()}")
    print(f"💾 Report θα αποθηκευτεί στο: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Βρες όλα τα .set αρχεία (και μέσα σε φακέλους)
    current_dir = os.getcwd()
    
    # Πρώτα δείξε τι υπάρχει στον φάκελο
    print(f"\n📋 Περιεχόμενα φακέλου:")
    all_items = os.listdir(current_dir)
    subject_folders = []
    other_items = []
    
    for item in all_items:
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path) and item.startswith('s') and item.endswith('.set'):
            subject_folders.append(item)
        elif os.path.isdir(item_path):
            other_items.append(f"📁 {item}/")
        else:
            other_items.append(f"📄 {item}")
    
    # Εμφάνιση των subject φακέλων πρώτα (sorted)
    subject_folders.sort()
    print(f"\n   📊 Subject Folders ({len(subject_folders)}):")
    for folder in subject_folders:
        print(f"      📁 {folder}/")
    
    if other_items:
        print(f"\n   📋 Άλλα αρχεία (πρώτα 10):")
        for item in other_items[:10]:
            print(f"      {item}")
        if len(other_items) > 10:
            print(f"      ... και άλλα {len(other_items) - 10}")
    
    set_files = find_set_files(current_dir)
    
    if not set_files:
        print("\n❌ Δεν βρέθηκαν .set αρχεία!")
        return None
    
    # Ταξινόμηση των αρχείων βάσει subject ID
    set_files.sort(key=lambda x: os.path.basename(x))
    
    print(f"\n📂 Βρέθηκαν {len(set_files)} αρχεία .set:")
    for i, f in enumerate(set_files, 1):
        subject_name = os.path.basename(f).replace('.set', '')
        print(f"   {i:2d}. {subject_name}")
    
    # Έλεγχος αν έχουμε τα αναμενόμενα 8 subjects
    expected_subjects = 8
    if len(set_files) == expected_subjects:
        print(f"\n✅ Τέλεια! Βρέθηκαν ακριβώς {expected_subjects} subjects όπως στο paper!")
    elif len(set_files) > expected_subjects:
        print(f"\n🎉 Εξαιρετικά! Βρέθηκαν {len(set_files)} subjects (περισσότερα από τα {expected_subjects} του paper)!")
    else:
        print(f"\n⚠️  Προσοχή: Βρέθηκαν μόνο {len(set_files)} subjects (αναμενόμενα: {expected_subjects})")
    
    # Φόρτωσε κάθε αρχείο
    loaded_data = {}
    events_summary = {}
    loading_errors = []
    
    for i, filepath in enumerate(set_files, 1):
        subject_name = os.path.basename(filepath).replace('.set', '')
        
        print(f"\n{'='*60}")
        print(f"ΦΟΡΤΩΣΗ SUBJECT {i}/{len(set_files)}: {subject_name}")
        print(f"{'='*60}")
        
        # Φόρτωση
        raw = load_eeg_file(filepath)
        
        if raw is not None:
            # Αποθήκευση
            loaded_data[subject_name] = raw
            
            # Εμφάνιση πληροφοριών
            events_found = show_eeg_info(raw, subject_name)
            events_summary[subject_name] = events_found
        else:
            loading_errors.append(subject_name)
    
    print(f"\n🎉 ΤΕΛΙΚΟ ΑΠΟΤΕΛΕΣΜΑ!")
    print("=" * 60)
    print(f"✅ Φορτώθηκαν επιτυχώς: {len(loaded_data)} από {len(set_files)} subjects")
    
    if loading_errors:
        print(f"❌ Αποτυχίες φόρτωσης: {len(loading_errors)}")
        for error_subject in loading_errors:
            print(f"   - {error_subject}")
    
    if loaded_data:
        print(f"\n📊 Αναλυτική Σύνοψη:")
        
        # Ταξινόμηση subjects για καλύτερη εμφάνιση
        sorted_subjects = sorted(loaded_data.keys())
        
        durations = []
        all_channels = []
        all_sfreqs = []
        
        for i, subject_name in enumerate(sorted_subjects, 1):
            raw = loaded_data[subject_name]
            duration_min = raw.times[-1] / 60
            n_channels = raw.info['nchan']
            sfreq = raw.info['sfreq']
            events_status = "✅" if events_summary.get(subject_name, False) else "❌"
            
            durations.append(duration_min)
            all_channels.append(n_channels)
            all_sfreqs.append(sfreq)
            
            print(f"   {i:2d}. {subject_name:15s}: {n_channels:2d}ch, {duration_min:5.1f}min, {sfreq:5.0f}Hz, Events:{events_status}")
        
        # Προχωρημένη στατιστική
        total_duration = sum(durations)
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        print(f"\n📈 Στατιστικά Dataset:")
        print(f"   📊 Συνολική διάρκεια: {total_duration:.1f} λεπτά ({total_duration/60:.1f} ώρες)")
        print(f"   📊 Μέση διάρκεια/subject: {avg_duration:.1f} ± {std_duration:.1f} λεπτά")
        print(f"   📊 Κανάλια: {min(all_channels)}-{max(all_channels)} (μέσος όρος: {np.mean(all_channels):.0f})")
        print(f"   📊 Sampling rates: {set(all_sfreqs)} Hz")
        print(f"   📊 Subjects με events: {sum(events_summary.values())}/{len(events_summary)} ({sum(events_summary.values())/len(events_summary)*100:.0f}%)")
        
        # Εκτίμηση μεγέθους dataset για epoching
        epochs_per_minute = 60  # 1-second epochs
        estimated_epochs = total_duration * epochs_per_minute
        print(f"\n🔮 Εκτιμήσεις για Epoching:")
        print(f"   📦 Εκτιμώμενα 1-sec epochs: ~{estimated_epochs:.0f}")
        print(f"   📦 Εκτιμώμενα epochs ανά κατηγορία: ~{estimated_epochs/2:.0f} Alert, ~{estimated_epochs/2:.0f} Fatigue")
        print(f"   💾 Εκτιμώμενο μέγεθος processed data: ~{estimated_epochs * 30 * 100 * 4 / (1024**3):.2f} GB")
        
        if sum(events_summary.values()) == 0:
            print(f"\n⚠️  ΣΗΜΑΝΤΙΚΟ: Δεν βρέθηκαν events σε κανένα subject!")
            print(f"   💡 Θα χρησιμοποιήσουμε time-based labeling:")
            print(f"   📅 Πρώτα 30 λεπτά = Alert state (label 0)")
            print(f"   😴 Τελευταία 30 λεπτά = Fatigue state (label 1)")
        else:
            print(f"\n✅ Εξαιρετικά! Βρέθηκαν events για event-based labeling!")
    
    print(f"\n📋 ΣΥΜΠΕΡΑΣΜΑΤΑ ΓΙΑ ΔΙΠΛΩΜΑΤΙΚΗ:")
    print("=" * 50)
    if len(loaded_data) >= 8:
        print(f"   ✅ {len(loaded_data)} subjects (ιδανικό για cross-validation)")
        print(f"   ✅ ~{total_duration:.0f} λεπτά EEG δεδομένων")
        print(f"   ✅ ~{estimated_epochs:.0f} epochs για training")
        print(f"   ✅ Ομοιογενή τεχνικά χαρακτηριστικά")
        print("\n📋 Επόμενα βήματα:")
        print("   1️⃣  Event processing & Epoching")
        print("   2️⃣  Feature extraction")
        print("   3️⃣  ESTCNN + baseline models")
        print("   4️⃣  Cross-validation evaluation")
    elif len(loaded_data) >= 5:
        print("✅ Καλό dataset για development:")
        print(f"   📊 {len(loaded_data)} subjects (αρκετά για αρχή)")
    else:
        print("⚠️  Περιορισμένο dataset:")
        print("   📝 Θα χρειαστούν περισσότερα subjects για robust evaluation")
    
    # Επιστροφή δεδομένων για περαιτέρω χρήση
    return loaded_data

# Εκτέλεση του script με αυτόματη αποθήκευση
if __name__ == "__main__":
    # Δημιουργία αρχείου output με timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"EEG_DataLoader_Report_{timestamp}.txt"
    output_filepath = os.path.join(OUTPUT_PATH, output_filename)
    
    # Έλεγχος αν υπάρχει ο φάκελος
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(f" Δημιουργήθηκε φάκελος: {OUTPUT_PATH}")
    
    print(f"🚀 Ξεκινάει EEG Data Loading με αυτόματη αποθήκευση...")
    print(f"📄 Output θα αποθηκευτεί: {output_filename}")
    print("=" * 60)
    
    # Καταγραφή όλου του output
    with OutputCapture(output_filepath):
        data = main()
    
    # Τελικό μήνυμα (εκτός αρχείου)
    print(f"\n{'='*60}")
    print(f"✅ ΟΛΟΚΛΗΡΩΣΗ ΕΠΙΤΥΧΟΥΣ!")
    print(f"📄 Report αποθηκεύτηκε: {output_filename}")
    print(f" Τοποθεσία: {OUTPUT_PATH}")
    print(f"{'='*60}")
    
    if data and len(data) >= 6:
        print("🎉 Dataset έτοιμο για επόμενο βήμα: Epoching & Feature Extraction!")
    else:
        print("📝 Ελέγξε το report για λεπτομέρειες και προσθήκη subjects.")



