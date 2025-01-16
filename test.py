import os
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw, DNS, NTP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from imblearn.over_sampling import SMOTE
import numpy as np
from datetime import datetime
from collections import defaultdict
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def extract_features_from_pcap(pcap_path):
    """
    Extract features from a PCAP file.
    """
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error reading file {pcap_path}: {e}")
        return None

    features = {
        "total_packets": 0,
        "total_bytes": 0,
        "unique_src_ips": set(),
        "unique_dst_ips": set(),
        "protocol_counts": {"TCP": 0, "UDP": 0, "ICMP": 0, "NTP": 0, "DNS": 0, "SSDP": 0, "IP": 0, "Other": 0},
        "http_get_count": 0,
        "http_post_count": 0,
        "icmp_echo_request": 0,
        "icmp_echo_reply": 0,
        "tcp_source_ports": set(),
        "tcp_dest_ports": set(),
        "tcp_flags": {},
        "udp_source_ports": set(),
        "udp_dest_ports": set(),
        "source_bytes": 0,
        "destination_bytes": 0,
        "time_differences": [],
        "connections_to_same_host": defaultdict(int),
        "invalid_sequence_count": 0
    }

    prev_time = None

    for packet in packets:
        features["total_packets"] += 1
        features["total_bytes"] += len(packet)

        if IP in packet:
            ip_layer = packet[IP]
            features["unique_src_ips"].add(ip_layer.src)
            features["unique_dst_ips"].add(ip_layer.dst)
            features["source_bytes"] += ip_layer.len
            features["connections_to_same_host"][ip_layer.dst] += 1
            features["protocol_counts"]["IP"] += 1

        if hasattr(packet, 'time'):
            curr_time = datetime.fromtimestamp(float(packet.time))
            if prev_time:
                time_diff = (curr_time - prev_time).total_seconds()
                features["time_differences"].append(time_diff)
            prev_time = curr_time

        if TCP in packet:
            features["protocol_counts"]["TCP"] += 1
            tcp_layer = packet[TCP]
            features["tcp_source_ports"].add(tcp_layer.sport)
            features["tcp_dest_ports"].add(tcp_layer.dport)
            features["destination_bytes"] += len(tcp_layer.payload)

            for flag in ["F", "S", "R", "P", "A", "U", "E", "C"]:
                if getattr(tcp_layer, "flags", 0) & getattr(TCP, flag, 0):
                    features["tcp_flags"].setdefault(flag, 0)
                    features["tcp_flags"][flag] += 1

            if tcp_layer.seq == 0:
                features["invalid_sequence_count"] += 1

            if Raw in packet:
                payload = packet[Raw].load.decode(errors="ignore")
                if payload.startswith("GET"):
                    features["http_get_count"] += 1
                elif payload.startswith("POST"):
                    features["http_post_count"] += 1

        elif UDP in packet:
            features["protocol_counts"]["UDP"] += 1
            udp_layer = packet[UDP]
            features["udp_source_ports"].add(udp_layer.sport)
            features["udp_dest_ports"].add(udp_layer.dport)
            features["destination_bytes"] += len(udp_layer.payload)

            if NTP in packet:
                features["protocol_counts"]["NTP"] += 1
            elif DNS in packet:
                features["protocol_counts"]["DNS"] += 1

        elif ICMP in packet:
            features["protocol_counts"]["ICMP"] += 1
            icmp_type = packet[ICMP].type
            if icmp_type == 8:
                features["icmp_echo_request"] += 1
            elif icmp_type == 0:
                features["icmp_echo_reply"] += 1

        elif Raw in packet and b"SSDP" in bytes(packet[Raw]):
            features["protocol_counts"]["SSDP"] += 1

        else:
            features["protocol_counts"]["Other"] += 1

    features["src_ip_count"] = len(features["unique_src_ips"])
    features["dst_ip_count"] = len(features["unique_dst_ips"])
    features["time_diff_variance"] = np.var(features["time_differences"]) if features["time_differences"] else 0
    features["connection_count"] = max(features["connections_to_same_host"].values()) if features["connections_to_same_host"] else 0

    for protocol in features["protocol_counts"]:
        features[protocol] = features["protocol_counts"][protocol]

    features["tcp_source_port_count"] = len(features["tcp_source_ports"])
    features["tcp_dest_port_count"] = len(features["tcp_dest_ports"])
    for flag, count in features["tcp_flags"].items():
        features[f"tcp_flag_{flag}"] = count

    features["udp_source_port_count"] = len(features["udp_source_ports"])
    features["udp_dest_port_count"] = len(features["udp_dest_ports"])

    del features["unique_src_ips"]
    del features["unique_dst_ips"]
    del features["protocol_counts"]
    del features["tcp_source_ports"]
    del features["tcp_dest_ports"]
    del features["tcp_flags"]
    del features["udp_source_ports"]
    del features["udp_dest_ports"]
    del features["time_differences"]
    del features["connections_to_same_host"]

    return features

def process_pcap_folder(folder_path, label):
    """
    Process a folder of PCAP files and extract features.
    """
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".pcap"):
            file_path = os.path.join(folder_path, file)
            features = extract_features_from_pcap(file_path)
            if features:
                features["label"] = label
                data.append(features)
    return pd.DataFrame(data)

# --- TRAINING CODE ---
# Define paths to datasets
benign_folder = "libcap-dataset/benign" 
ddos_folder = "libcap-dataset/ddos"

# Process datasets
benign_data = process_pcap_folder(benign_folder, label=0)
ddos_data = process_pcap_folder(ddos_folder, label=1)

# Combine datasets
data = pd.concat([benign_data, ddos_data], ignore_index=True)

desc_stats = data.describe()

# Print the descriptive statistics table
print(desc_stats)

desc_stats = desc_stats.transpose()

data.hist(bins=150, figsize=(15, 15))
plt.show()

# Extract features and labels
X = data.drop(columns=["label"])
y = data["label"]

# Save feature names for later alignment
feature_names = X.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Use SMOTE for balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Split the balanced dataset
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Train an SVM model
model = SVC(kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "svm_model_smote.pkl")

# Save a scaled subset of X_train for LIME (debugging or testing purposes)
train_sample = pd.DataFrame(X_train[:10], columns=feature_names)
joblib.dump(train_sample, "lime_train_sample.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
