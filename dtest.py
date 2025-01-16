import os
from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

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
        "protocol_counts": {"TCP": 0, "UDP": 0, "ICMP": 0, "Other": 0}
    }

    for packet in packets:
        features["total_packets"] += 1
        features["total_bytes"] += len(packet)

        if IP in packet:
            ip_layer = packet[IP]
            features["unique_src_ips"].add(ip_layer.src)
            features["unique_dst_ips"].add(ip_layer.dst)

        if TCP in packet:
            features["protocol_counts"]["TCP"] += 1
        elif UDP in packet:
            features["protocol_counts"]["UDP"] += 1
        elif ICMP in packet:
            features["protocol_counts"]["ICMP"] += 1
        else:
            features["protocol_counts"]["Other"] += 1

    features["src_ip_count"] = len(features["unique_src_ips"])
    features["dst_ip_count"] = len(features["unique_dst_ips"])

    # Flatten protocol counts
    for protocol in features["protocol_counts"]:
        features[protocol] = features["protocol_counts"][protocol]

    del features["unique_src_ips"]
    del features["unique_dst_ips"]
    del features["protocol_counts"]

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

# Define paths to datasets
benign_folder = "libcap-dataset/benign"
ddos_folder = "libcap-dataset/ddos"

# Process datasets
benign_data = process_pcap_folder(benign_folder, label=0)
ddos_data = process_pcap_folder(ddos_folder, label=1)

# Combine datasets
data = pd.concat([benign_data, ddos_data], ignore_index=True)

# Prepare data for training
X = data.drop(columns=["label"])
y = data["label"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build the deep learning model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Set up TensorBoard callback
log_dir = "logs/fit"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Save the model architecture as an image
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

# Train the model with TensorBoard callback
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[tensorboard_callback])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Print classification report
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))

# Save the model
model.save("pcap_classifier_model.h5")
