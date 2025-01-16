import os
import joblib
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw, DNS, NTP
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime
import numpy as np
from collections import defaultdict

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

def classify_pcap(pcap_path, model_path, scaler_path, feature_names_path, lime_train_sample_path):
    """
    Classify a single PCAP file and explain the results using LIME.
    """
    try:
        # Load model and preprocessing artifacts
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        lime_train_sample = joblib.load(lime_train_sample_path)

        # Extract features from the PCAP file
        features = extract_features_from_pcap(pcap_path)
        if features is None:
            print("No features extracted from the PCAP file.")
            return

        # Align extracted features with training features
        feature_vector = [features.get(feature, 0) for feature in feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(feature_vector)

        # Predict the class
        prediction = model.predict(scaled_features)[0]
        prediction_probabilities = model.predict_proba(scaled_features)[0][prediction]

        print(f"Prediction: {'Benign' if prediction == 0 else 'DDoS'}")
        print(f"Prediction Probabilities: {prediction_probabilities * 100:.2f}%")

        # LIME explanation
        explainer = LimeTabularExplainer(
            training_data=lime_train_sample.values,
            feature_names=feature_names,
            class_names=["Benign", "DDoS"],
            mode="classification"
        )
        explanation = explainer.explain_instance(
            data_row=scaled_features[0],
            predict_fn=model.predict_proba
        )

        # Save the explanation to an HTML file
        explanation_file = f"lime_explanation_{os.path.basename(pcap_path).replace('.pcap', '.html')}"
        explanation.save_to_file(explanation_file)
        print(f"LIME explanation saved to {explanation_file}.")
    except Exception as e:
        print(f"Error during classification: {e}")

# --- Example usage ---
pcap_file = "example.pcap"
classify_pcap(
    pcap_path=pcap_file,
    model_path="svm_model_smote.pkl",
    scaler_path="scaler.pkl",
    feature_names_path="feature_names.pkl",
    lime_train_sample_path="lime_train_sample.pkl"
)

