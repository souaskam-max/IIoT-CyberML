"""
IIoT-CyberML: A Multi-Dataset Detection System - REVIEWER VERSION
Authors: Kamal Souadih and Foudil Mir

REVIEWER NOTE:
- This script contains 100% of the original code for full reproducibility
- All other outputs (CSV, JSON) are saved for evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import time
import os
import platform
import psutil
import threading
import gc
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, roc_curve, auc, roc_auc_score, 
                             recall_score, f1_score)
from sklearn.utils import shuffle
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Import ML models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for resource monitoring
resource_logs = []
monitoring_active = False

# ============================================
# RESOURCE MONITORING FUNCTIONS
# ============================================

def get_gpu_info():
    """Get GPU information using nvidia-smi if available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    # REVIEWER VERSION: Anonymize GPU name
                    gpu_info.append({'name': 'NVIDIA_GPU_ANONYMIZED', 'memory_mb': float(parts[1].strip())})
            return gpu_info
    except:
        pass
    return None

def get_gpu_usage():
    """Get current GPU memory usage and utilization"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_stats.append({
                        'memory_used_mb': float(parts[0].strip()),
                        'utilization_percent': float(parts[1].strip())
                    })
            return gpu_stats[0] if gpu_stats else {'memory_used_mb': 0, 'utilization_percent': 0}
    except:
        pass
    return {'memory_used_mb': 0, 'utilization_percent': 0}

def monitor_resources(interval=2.0):
    """Background thread to monitor system resources"""
    global resource_logs, monitoring_active
    
    process = psutil.Process()
    
    while monitoring_active:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            ram_used_mb = ram_info.used / (1024**2)
            
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024**2)
            disk_write_mb = disk_io.write_bytes / (1024**2)
            
            gpu_stats = get_gpu_usage()
            
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'ram_used_mb': ram_used_mb,
                'disk_read_mb': disk_read_mb,
                'disk_write_mb': disk_write_mb,
                'gpu_mem_used_mb': gpu_stats['memory_used_mb'],
                'gpu_util_percent': gpu_stats['utilization_percent']
            }
            
            resource_logs.append(log_entry)
            time.sleep(interval)
            
        except Exception as e:
            logging.error(f"Error monitoring resources: {e}")
            break

def start_monitoring():
    """Start resource monitoring in background thread"""
    global monitoring_active, resource_logs
    monitoring_active = True
    resource_logs = []
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    return monitor_thread

def stop_monitoring():
    """Stop resource monitoring"""
    global monitoring_active
    monitoring_active = False
    time.sleep(2.5)

def collect_environment_metadata(seed=42, batch_size=32):
    """Collect comprehensive environment metadata"""
    metadata = {
        'date': datetime.now().strftime('%Y-%m'),  # REVIEWER: Month-level precision only
        'machine_name': 'ANONYMIZED_FOR_REVIEW',   # REVIEWER: Anonymized
        'os': f"{platform.system()} {platform.release()}",
        'cpu': {
            'model': 'x86_64_ANONYMIZED',  # REVIEWER: Anonymized
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
        },
        'ram_total_gb': round(psutil.virtual_memory().total / (1024**3)),  # Rounded
        'seed': seed,
        'batch_size': batch_size
    }
    
    gpu_info = get_gpu_info()
    if gpu_info:
        metadata['gpu'] = gpu_info
    else:
        metadata['gpu'] = 'Not available'
    
    import sys
    import sklearn
    metadata['versions'] = {
        'python': sys.version.split()[0],  # Version number only
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit-learn': sklearn.__version__,
    }
    
    try:
        import xgboost
        metadata['versions']['xgboost'] = xgboost.__version__
    except:
        pass
    
    return metadata

def measure_inference_latency(model, X_sample, n_runs=5, batch_sizes=[1, 32], device='CPU'):
    """Measure inference latency and throughput"""
    results = []
    
    for batch_size in batch_sizes:
        latencies = []
        ram_usage = []
        gpu_usage = []
        
        # Warm-up
        for _ in range(2):
            if batch_size == 1:
                _ = model.predict(X_sample[:1])
            else:
                _ = model.predict(X_sample[:min(batch_size, len(X_sample))])
        
        # Actual measurements
        for run in range(n_runs):
            ram_before = psutil.virtual_memory().used / (1024**2)
            gpu_stats_before = get_gpu_usage()
            
            start_time = time.perf_counter()
            if batch_size == 1:
                _ = model.predict(X_sample[:1])
            else:
                _ = model.predict(X_sample[:min(batch_size, len(X_sample))])
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            ram_after = psutil.virtual_memory().used / (1024**2)
            gpu_stats_after = get_gpu_usage()
            
            ram_usage.append(ram_after - ram_before)
            gpu_usage.append(gpu_stats_after['memory_used_mb'] - gpu_stats_before['memory_used_mb'])
        
        latencies_np = np.array(latencies)
        throughput = (batch_size * 1000) / np.mean(latencies)
        
        result = {
            'device': device,
            'batch_size': batch_size,
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_p50_ms': float(np.percentile(latencies, 50)),
            'latency_p95_ms': float(np.percentile(latencies, 95)),
            'latency_max_ms': float(np.max(latencies)),
            'throughput_samples_per_s': float(throughput),
            'ram_used_mb_during_inference': float(np.mean(ram_usage)) if ram_usage else 0,
            'gpu_mem_used_mb_during_inference': float(np.mean(gpu_usage)) if gpu_usage else 0
        }
        
        results.append(result)
        logging.info(f"Latency - Device: {device}, Batch: {batch_size}, Mean: {result['latency_mean_ms']:.2f}ms")
    
    return results

# Color palette definition for visualization
n_colors = 20
colors = plt.cm.get_cmap('viridis', n_colors)
cmap = ListedColormap(colors(np.linspace(0, 1, n_colors)))
dataset_colors = {1: '\033[91m', 2: '\033[94m', 3: '\033[92m', 4: '\033[93m', 'reset': '\033[0m'}
dataset_plot_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}

# ---------------------------
# Data preparation functions
# ---------------------------
def prepare_data_1(path_csv, nrows=100000):
    logging.info("Loading and preparing data from Dataset1...")
    data = pd.read_csv(path_csv, nrows=nrows)
    data = pd.get_dummies(data, columns=['Protocol', 'Service'])
    y = data['class1']
    data = data.drop(['Date', 'Timestamp', 'Scr_IP', 'Des_IP', 'class1', 'class2', 'class3'], axis=1)
    data = data.replace({False: 0, 'FALSE': 0, 'false': 0, True: 1, 'TRUE': 1, 'true': 1,
                         '-': np.nan, '?': np.nan, '': np.nan, ' ': np.nan}).replace({'[A-Za-z]': np.nan}, regex=True)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(thresh=int(0.7 * len(data)), axis=1)
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    class_labels1 = dict(zip(y_int, y))
    class_dataset_map1 = {label: 1 for label in y.unique()}
    data = data.astype('float32')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(scaled, columns=data.columns)
    return df_scaled, y_int, le, scaler, class_labels1, class_dataset_map1

def remove_outliers(df, threshold=1.5):
    if 'Attack_type' in df.columns:
        groups = []
        for cl in df['Attack_type'].unique():
            group = df[df['Attack_type'] == cl]
            numeric_cols = group.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = group[col].quantile(0.25)
                Q3 = group[col].quantile(0.75)
                IQR = Q3 - Q1
                group = group[(group[col] >= Q1 - threshold * IQR) & (group[col] <= Q3 + threshold * IQR)]
            groups.append(group)
        return pd.concat(groups)
    return df

def add_rolling_features(df, window=3):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[f'{col}_roll_mean'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_roll_std'] = df[col].rolling(window=window, min_periods=1).std().fillna(0)
    return df

def prepare_data_2(path_csv):
    logging.info("Loading and preparing data from Dataset2...")
    df = pd.read_csv(path_csv, low_memory=False)
    cols_to_drop = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
                    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
                    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
                    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
    df = df.drop(columns=cols_to_drop, errors='ignore').dropna().drop_duplicates()
    df = remove_outliers(df, threshold=1.5)
    df = add_rolling_features(df, window=5)
    categorical_cols = ['http.request.method', 'http.referer', 'http.request.version',
                        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']
    for col in categorical_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col])
    le = LabelEncoder()
    y = le.fit_transform(df['Attack_type'])
    class_labels2 = dict(zip(y, df['Attack_type']))
    class_dataset_map2 = {label: 2 for label in df['Attack_type'].unique()}
    X_df = df.drop(columns=['Attack_type'])
    # BUGFIX: Select numeric columns before conversion to avoid IP address errors
    X_df = X_df.select_dtypes(include=[np.number])
    X_df = X_df.astype('float32')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_df)
    df_scaled = pd.DataFrame(scaled, columns=X_df.columns)
    return df_scaled, y, scaler, np.unique(df['Attack_type']), class_labels2, class_dataset_map2

def prepare_data_3(path_csv):
    logging.info("Loading and preparing data from Dataset3...")
    data = pd.read_csv(path_csv)
    remove_cols = ['StartTime', 'LastTime', 'SrcAddr', 'DstAddr', 'sIpId', 'dIpId']
    data = data.drop(remove_cols + ['Target'], axis=1)
    data.drop_duplicates(inplace=True)
    y = data['Traffic']
    data = data.drop(columns=['Traffic'])
    logging.info("Selecting numeric columns in Dataset3...")
    data = data.select_dtypes(include=[np.number])
    logging.info("Filling missing values with median in Dataset3...")
    data.fillna(data.median(), inplace=True)
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    class_labels3 = dict(zip(y_int, y))
    class_dataset_map3 = {label: 3 for label in y.unique()}
    data = data.astype('float32')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(scaled, columns=data.columns)
    return df_scaled, y_int, le, scaler, class_labels3, class_dataset_map3

def prepare_data_4(path_csv):
    logging.info("Loading and preparing data from Dataset4...")
    data = pd.read_csv(path_csv)
    data = data.drop(columns=["ts", "src_ip", "src_port", "dst_port", "dst_ip"])
    data = data.replace("-", np.NaN).apply(lambda x: x.fillna(x.value_counts().index[0]))
    data.drop_duplicates(inplace=True)
    y = data['type']
    data = data.drop(columns=['type', 'label'])
    logging.info("Selecting numeric columns in Dataset4...")
    data = data.select_dtypes(include=[np.number])
    logging.info("Filling missing values with median in Dataset4...")
    data.fillna(data.median(), inplace=True)
    le = LabelEncoder()
    y_int = le.fit_transform(y)
    class_labels4 = dict(zip(y_int, y))
    class_dataset_map4 = {label: 4 for label in y.unique()}
    data = data.astype('float32')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    df_scaled = pd.DataFrame(scaled, columns=data.columns)
    return df_scaled, y_int, le, scaler, class_labels4, class_dataset_map4

# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def identify_minority_classes(y, threshold_percentage=0.01):
    class_counts = pd.Series(y).value_counts()
    total_samples = len(y)
    minority_classes = class_counts[class_counts / total_samples < threshold_percentage].index
    return minority_classes

def colored_classification_report(report, class_dataset_mapping, all_classes_labels_encoded_to_name):
    lines = report.split('\n')
    colored_lines = []
    for line in lines:
        if not line.strip():
            colored_lines.append(line)
            continue
        parts = line.split()
        if not parts:
            colored_lines.append(line)
            continue
        try:
            class_name_candidate = parts[0]
            encoded_value = None
            for enc, name in all_classes_labels_encoded_to_name.items():
                if name == class_name_candidate:
                    encoded_value = enc
                    break
            if encoded_value is None and len(parts) > 1:
                class_name_candidate = parts[1]
                for enc, name in all_classes_labels_encoded_to_name.items():
                    if name == class_name_candidate:
                        encoded_value = enc
                        break
            if encoded_value is not None:
                original_class_name = all_classes_labels_encoded_to_name[encoded_value]
                dataset_id = None
                for name, ds_id in class_dataset_mapping.items():
                    if name == original_class_name:
                        dataset_id = ds_id
                        break
                if dataset_id:
                    colored_class_name = f"{dataset_colors[dataset_id]}{original_class_name}{dataset_colors['reset']}"
                    colored_line = line.replace(original_class_name, colored_class_name, 1)
                    colored_lines.append(colored_line)
                else:
                    colored_lines.append(line)
            else:
                colored_lines.append(line)
        except Exception as e:
            colored_lines.append(line)
    return '\n'.join(colored_lines)

# ---------------------------
# Main function
# ---------------------------
def main_unified():
    # Output directory
    output_dir = Path('../results')
    output_dir.mkdir(exist_ok=True)
    
    # ========================================
    # 1) COLLECT ENVIRONMENT METADATA
    # ========================================
    logging.info("=" * 60)
    logging.info("COLLECTING ENVIRONMENT METADATA")
    logging.info("=" * 60)
    
    SEED = 42
    BATCH_SIZE = 32
    
    env_metadata = collect_environment_metadata(seed=SEED, batch_size=BATCH_SIZE)
    with open(output_dir / 'env_metadata.json', 'w') as f:
        json.dump(env_metadata, f, indent=2)
    
    logging.info(f"Environment metadata saved (anonymized for review)")
    
    # ========================================
    # DATA LOADING
    # ========================================
    PATH1 = '/kaggle/input/xiiotid-iiot-intrusion-dataset/X-IIoTID dataset.csv'
    PATH2 = '/kaggle/input/edgeiiotset-cyber-security-dataset-of-iot-iiot/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv'
    PATH3 = '/kaggle/input/wustl-iiot-2021/wustl_iiot_2021.csv'
    PATH4 = '/kaggle/input/ton-iot-train-test/TON_IoT_Train_Test_Network.csv'

    # Data Preparation
    df1, y1, le1, scaler1, labels1, class_dataset_map1 = prepare_data_1(PATH1)
    ds1_labels = np.full(df1.shape[0], 1)
    df2, y2, scaler2, classes2_arr, labels2, class_dataset_map2 = prepare_data_2(PATH2)
    ds2_labels = np.full(df2.shape[0], 2)
    df3, y3, le3, scaler3, labels3, class_dataset_map3 = prepare_data_3(PATH3)
    ds3_labels = np.full(df3.shape[0], 3)
    df4, y4, le4, scaler4, labels4, class_dataset_map4 = prepare_data_4(PATH4)
    ds4_labels = np.full(df4.shape[0], 4)

    # Label Harmonization
    unique_combined_labels_names = np.unique(np.hstack((list(labels1.values()), list(labels2.values()),
                                                         list(labels3.values()), list(labels4.values()))))
    label_encoder_combined = LabelEncoder().fit(unique_combined_labels_names)
    y1_encoded = np.array([label_encoder_combined.transform([labels1[val]])[0] for val in y1])
    y2_encoded = np.array([label_encoder_combined.transform([labels2[val]])[0] for val in y2])
    y3_encoded = np.array([label_encoder_combined.transform([labels3[val]])[0] for val in y3])
    y4_encoded = np.array([label_encoder_combined.transform([labels4[val]])[0] for val in y4])
    all_classes_encoded = np.unique(np.hstack((y1_encoded, y2_encoded, y3_encoded, y4_encoded)))
    all_classes_labels_encoded_to_name = {i: label_encoder_combined.inverse_transform([i])[0] for i in all_classes_encoded}
    class_dataset_mapping = {}
    for d in [class_dataset_map1, class_dataset_map2, class_dataset_map3, class_dataset_map4]:
        for label, ds_id in d.items():
            if label not in class_dataset_mapping:
                class_dataset_mapping[label] = ds_id
    
    # Feature Harmonization
    logging.info("Harmonizing features using UNION followed by MEDIAN IMPUTATION...")
    all_columns = df1.columns.union(df2.columns).union(df3.columns).union(df4.columns)
    logging.info(f"Total unique features: {len(all_columns)}")

    df1_aligned = df1.reindex(columns=all_columns, fill_value=np.nan)
    df2_aligned = df2.reindex(columns=all_columns, fill_value=np.nan)
    df3_aligned = df3.reindex(columns=all_columns, fill_value=np.nan)
    df4_aligned = df4.reindex(columns=all_columns, fill_value=np.nan)

    X_combined_df = pd.concat([df1_aligned, df2_aligned, df3_aligned, df4_aligned], axis=0)
    y_combined = np.hstack((y1_encoded, y2_encoded, y3_encoded, y4_encoded))
    dataset_combined = np.hstack((ds1_labels, ds2_labels, ds3_labels, ds4_labels))

    logging.info("Applying SimpleImputer with median strategy...")
    imputer = SimpleImputer(strategy='median')
    X_combined = imputer.fit_transform(X_combined_df)
    X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=1e10, neginf=-1e10)

    # ========================================
    # 2) DATASET STATISTICS
    # ========================================
    logging.info("COLLECTING DATASET STATISTICS")
    
    class_counts_dict = {}
    for class_idx in np.unique(y_combined):
        class_counts_dict[all_classes_labels_encoded_to_name[class_idx]] = int(np.sum(y_combined == class_idx))
    
    dataset_stats = {
        'n_samples_total': int(len(X_combined)),
        'n_features': int(X_combined.shape[1]),
        'n_classes': int(len(np.unique(y_combined))),
        'class_counts': class_counts_dict
    }
    
    with open(output_dir / 'dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    # Shuffle and Train/Test Split
    X_combined, y_combined, dataset_combined = shuffle(X_combined, y_combined, dataset_combined, random_state=42)
    X_train_combined, X_test_combined, y_train_combined, y_test_combined, dataset_train, dataset_test = train_test_split(
        X_combined, y_combined, dataset_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    minority_classes_train_encoded = identify_minority_classes(y_train_combined)
    minority_classes_train_names = [all_classes_labels_encoded_to_name[i] for i in minority_classes_train_encoded]
    print("\nMinority classes:", minority_classes_train_names)

    unique_classes_test = np.unique(y_test_combined)
    target_names_map = {i: all_classes_labels_encoded_to_name[i] for i in unique_classes_test}
    sorted_unique_classes = sorted(target_names_map.keys())
    target_names = [f"{target_names_map[i]}" for i in sorted_unique_classes]
    target_names_with_idx = [f"{i} ({target_names_map[i]})" for i in sorted_unique_classes]

    # =================================================================================
    # 3) MODEL TRAINING WITH RESOURCE MONITORING
    # =================================================================================
    logging.info("=" * 60)
    logging.info("STARTING MODEL TRAINING")
    logging.info("=" * 60)
    
    # Start monitoring
    start_monitoring()
    
    training_start_time = time.time()
    model_results = {}
    trained_models = {}
    
    # Model Training
    models = {
        'XGBoost': XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                                 use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42)
    }

    for name, model in models.items():
        logging.info(f"Training {name} model...")
        epoch_start = time.time()
        
        model.fit(X_train_combined, y_train_combined)
        
        epoch_time = time.time() - epoch_start
        trained_models[name] = model
        y_pred = model.predict(X_test_combined)
        
        # Calculate metrics
        acc = accuracy_score(y_test_combined, y_pred)
        prec = precision_score(y_test_combined, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_combined, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_combined, y_pred, average='weighted', zero_division=0)
        
        model_results[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'train_time_s': epoch_time
        }

        print(f"\n--- {name} Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        report = classification_report(y_test_combined, y_pred, target_names=target_names_with_idx, zero_division=0)
        print(colored_classification_report(report, class_dataset_mapping, all_classes_labels_encoded_to_name))
        
        gc.collect()

    # VotingClassifier
    voting_clf = VotingClassifier(estimators=[(name, model) for name, model in models.items()], voting='soft')
    logging.info("Training VotingClassifier model...")
    ensemble_start = time.time()
    
    voting_clf.fit(X_train_combined, y_train_combined)
    
    ensemble_time = time.time() - ensemble_start
    trained_models['VotingClassifier'] = voting_clf
    y_pred_voting = voting_clf.predict(X_test_combined)
    
    acc_voting = accuracy_score(y_test_combined, y_pred_voting)
    prec_voting = precision_score(y_test_combined, y_pred_voting, average='weighted', zero_division=0)
    rec_voting = recall_score(y_test_combined, y_pred_voting, average='weighted', zero_division=0)
    f1_voting = f1_score(y_test_combined, y_pred_voting, average='weighted', zero_division=0)
    
    model_results['VotingClassifier'] = {
        'accuracy': acc_voting,
        'precision': prec_voting,
        'recall': rec_voting,
        'f1_score': f1_voting,
        'train_time_s': ensemble_time
    }

    print(f"\n--- VotingClassifier Evaluation ---")
    print(f"Accuracy: {acc_voting:.4f}")
    print(f"Precision: {prec_voting:.4f}")
    print(f"Recall: {rec_voting:.4f}")
    print(f"F1-Score: {f1_voting:.4f}")
    
    report_voting = classification_report(y_test_combined, y_pred_voting, target_names=target_names_with_idx, zero_division=0)
    print(colored_classification_report(report_voting, class_dataset_mapping, all_classes_labels_encoded_to_name))
    
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])

    total_training_time = time.time() - training_start_time
    
    # Stop monitoring
    stop_monitoring()
    
    # ========================================
    # 4) SAVE RESOURCE MONITORING DATA
    # ========================================
    logging.info("Saving resource monitoring data...")
    
    resource_df = pd.DataFrame(resource_logs)
    if len(resource_df) > 0:
        resource_df['timestamp'] = pd.to_datetime(resource_df['timestamp'])
        resource_df.to_csv(output_dir / 'train_resource.csv', index=False)
        
        peak_ram_mb = float(resource_df['ram_used_mb'].max())
        peak_gpu_mem_mb = float(resource_df['gpu_mem_used_mb'].max())
        avg_cpu = float(resource_df['cpu_percent'].mean())
    else:
        peak_ram_mb = 0
        peak_gpu_mem_mb = 0
        avg_cpu = 0
    
    # ========================================
    # 5) MODEL INFO
    # ========================================
    logging.info("COLLECTING MODEL INFORMATION")
    
    # Save best model
    import pickle
    best_model = trained_models[best_model_name]
    
    # REVIEWER VERSION: Model weights NOT saved
    logging.info("=" * 80)
    logging.info("REVIEWER NOTE: Model weights (.pkl) NOT saved")
    logging.info("Full model will be published on Zenodo post-acceptance")
    logging.info("=" * 80)
    
    # model_file = output_dir / 'best_model.pkl'  # COMMENTED FOR REVIEWER
    # with open(model_file, 'wb') as f:            # COMMENTED FOR REVIEWER
    #     pickle.dump(best_model, f)               # COMMENTED FOR REVIEWER
    
    # Estimate model size without saving
    import io
    buffer = io.BytesIO()
    pickle.dump(best_model, buffer)
    model_size_mb = len(buffer.getvalue()) / (1024**2)
    
    # Count parameters
    n_parameters = 0
    if hasattr(best_model, 'estimators_'):
        for estimator in best_model.estimators_:
            if hasattr(estimator, 'tree_'):
                n_parameters += estimator.tree_.node_count
            elif hasattr(estimator, 'estimators_'):
                for tree in estimator.estimators_:
                    if hasattr(tree, 'tree_'):
                        n_parameters += tree.tree_.node_count
    elif hasattr(best_model, 'get_booster'):
        n_parameters = len(best_model.get_booster().get_dump())
    
    model_info = {
        'model_name': best_model_name,
        'model_size_mb': float(model_size_mb),
        'n_parameters': int(n_parameters),
        'architecture': str(type(best_model).__name__),
        'note': 'Model weights excluded from reviewer submission - will be published on Zenodo post-acceptance'
    }
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # ========================================
    # 6) INFERENCE LATENCY MEASUREMENTS
    # ========================================
    logging.info("MEASURING INFERENCE LATENCY")
    
    inference_sample = X_test_combined[:1000]
    
    latency_results = measure_inference_latency(
        best_model, inference_sample, 
        n_runs=5, 
        batch_sizes=[1, 32], 
        device='GPU' if 'XGBoost' in best_model_name else 'CPU'
    )
    
    latency_df = pd.DataFrame(latency_results)
    latency_df.to_csv(output_dir / 'inference_latency.csv', index=False)
    
    # ========================================
    # 7) MINORITY CLASS EXPERIMENTS
    # ========================================
    logging.info("MINORITY CLASS EXPERIMENTS")
    
    minority_results_all = []
    
    for minority_class in minority_classes_train_encoded:
        if minority_class in unique_classes_test:
            mask = (y_test_combined == minority_class)
            if mask.sum() == 0:
                continue
            
            y_true_class = y_test_combined[mask]
            y_pred_class = y_pred_voting[mask]
            
            y_true_binary = (y_true_class == minority_class).astype(int)
            y_pred_binary = (y_pred_class == minority_class).astype(int)
            
            recall = recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            precision = precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
            
            minority_results_all.append({
                'class': all_classes_labels_encoded_to_name[minority_class],
                'recall': float(recall),
                'precision': float(precision),
                'f1_score': float(f1),
                'support': int(mask.sum()),
                'method': 'Baseline'
            })
    
    baseline_macro_f1 = f1_score(y_test_combined, y_pred_voting, average='macro', zero_division=0)
    
    logging.info(f"Baseline Macro-F1: {baseline_macro_f1:.4f}")
    
    minority_df = pd.DataFrame(minority_results_all)
    minority_df.to_csv(output_dir / 'minority_results.csv', index=False)
    
    # ========================================
    # 8) CROSS-DATASET EVALUATION
    # ========================================
    logging.info("CROSS-DATASET EVALUATION")
    
    cross_domain_results = []
    
    macro_f1 = f1_score(y_test_combined, y_pred_voting, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test_combined, y_pred_voting, average='micro', zero_division=0)
    recall_macro = recall_score(y_test_combined, y_pred_voting, average='macro', zero_division=0)
    
    cross_domain_results.append({
        'train_dataset': 'Fused',
        'test_dataset': 'Combined',
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'recall_macro': float(recall_macro),
        'train_time_s': float(total_training_time)
    })
    
    cross_domain_df = pd.DataFrame(cross_domain_results)
    cross_domain_df.to_csv(output_dir / 'cross_domain_results.csv', index=False)
    
    # ========================================
    # 9) IIOT SUITABILITY ANALYSIS
    # ========================================
    logging.info("ANALYZING IIOT IDS SUITABILITY")
    
    iiot_suitability = {
        'latency_analysis': {
            'batch_1_mean_ms': float(latency_df[latency_df['batch_size'] == 1]['latency_mean_ms'].values[0]),
            'batch_1_p95_ms': float(latency_df[latency_df['batch_size'] == 1]['latency_p95_ms'].values[0]),
            'real_time_capable': bool(latency_df[latency_df['batch_size'] == 1]['latency_p95_ms'].values[0] < 100),
            'throughput_samples_per_s': float(latency_df[latency_df['batch_size'] == 1]['throughput_samples_per_s'].values[0])
        },
        'resource_efficiency': {
            'peak_ram_gb': float(peak_ram_mb / 1024),
            'peak_gpu_mb': float(peak_gpu_mem_mb),
            'avg_cpu_percent': float(avg_cpu),
            'model_size_mb': float(model_size_mb),
            'edge_deployable': bool(model_size_mb < 100 and peak_ram_mb < 2048),
            'embedded_friendly': bool(model_size_mb < 50)
        },
        'detection_performance': {
            'overall_accuracy': float(model_results[best_model_name]['accuracy']),
            'macro_f1': float(baseline_macro_f1),
            'micro_f1': float(micro_f1),
            'minority_class_recall_avg': float(np.mean([r['recall'] for r in minority_results_all])) if minority_results_all else 0,
            'balanced_performance': bool(baseline_macro_f1 > 0.7)
        },
        'operational_metrics': {
            'training_time_hours': float(total_training_time / 3600),
            'samples_processed': int(len(X_train_combined) + len(X_test_combined)),
            'n_features': int(len(all_columns)),
            'n_classes': int(len(unique_classes_test)),
            'training_samples_per_second': float(len(X_train_combined) / total_training_time)
        },
        'iiot_readiness_score': 0.0
    }
    
    # Calculate IIoT readiness score
    score = 0
    
    if iiot_suitability['latency_analysis']['batch_1_p95_ms'] < 10:
        score += 30
    elif iiot_suitability['latency_analysis']['batch_1_p95_ms'] < 50:
        score += 20
    elif iiot_suitability['latency_analysis']['batch_1_p95_ms'] < 100:
        score += 10
    
    if iiot_suitability['resource_efficiency']['edge_deployable']:
        score += 15
    if iiot_suitability['resource_efficiency']['embedded_friendly']:
        score += 10
    
    acc_score = iiot_suitability['detection_performance']['overall_accuracy'] * 20
    f1_score_val = baseline_macro_f1 * 15
    score += acc_score + f1_score_val
    
    if iiot_suitability['detection_performance']['balanced_performance']:
        score += 10
    
    iiot_suitability['iiot_readiness_score'] = float(min(100, score))
    
    if score >= 80:
        suitability_level = "EXCELLENT - Highly suitable for IIoT IDS deployment"
    elif score >= 60:
        suitability_level = "GOOD - Suitable for IIoT IDS with minor optimizations"
    elif score >= 40:
        suitability_level = "MODERATE - Suitable for cloud-based IIoT IDS"
    else:
        suitability_level = "LIMITED - Requires significant optimization"
    
    iiot_suitability['suitability_assessment'] = suitability_level
    
    with open(output_dir / 'iiot_suitability.json', 'w') as f:
        json.dump(iiot_suitability, f, indent=2)
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("🎯 IIOT IDS ANALYSIS COMPLETE (REVIEWER VERSION)")
    print("=" * 80)
    print(f"\n📊 Key Results:")
    print(f"   • Best Model: {best_model_name}")
    print(f"   • Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
    print(f"   • Macro-F1: {baseline_macro_f1:.4f}")
    print(f"   • IIoT Readiness Score: {iiot_suitability['iiot_readiness_score']:.1f}/100")
    print(f"\n⚡ Performance Metrics:")
    print(f"   • Inference Latency (P95): {iiot_suitability['latency_analysis']['batch_1_p95_ms']:.2f} ms")
    print(f"   • Real-time Capable: {iiot_suitability['latency_analysis']['real_time_capable']}")
    print(f"\n📁 Results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("✨ Ready and finished")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main_unified()
