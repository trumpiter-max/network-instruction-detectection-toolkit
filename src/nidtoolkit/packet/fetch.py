from scapy.all import rdpcap, IP, TCP, UDP, ICMP
import pandas as pd
from collections import Counter
import math
import os

def calculate_entropy(counter):
    """Calculate Shannon entropy for a given Counter object."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    entropy = -sum((count / total) * math.log2(count / total) for count in counter.values())
    return entropy

def extract_features_from_file(pcap_path: str):
    # Load the pcap file
    packets = rdpcap(pcap_path)

    # Initialize counters and features
    feature_dict = {
        'total_packets': 0,
        'total_bytes': 0,
        'unique_src_ips': set(),
        'unique_dst_ips': set(),
        'packet_sizes': [],
        'tcp_flags': {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0},
        'src_ports': Counter(),
        'dst_ports': Counter(),
        'protocols': Counter()
    }

    # Process the packets
    for packet in packets:
        feature_dict['total_packets'] += 1
        feature_dict['total_bytes'] += len(packet)
        feature_dict['packet_sizes'].append(len(packet))

        # Check for IP layer and extract IP addresses
        if packet.haslayer(IP):
            ip_layer = packet[IP]
            feature_dict['unique_src_ips'].add(ip_layer.src)
            feature_dict['unique_dst_ips'].add(ip_layer.dst)

            # Count protocols
            feature_dict['protocols'][ip_layer.proto] += 1

        # Check for TCP layer and extract flags and ports
        if packet.haslayer(TCP):
            tcp_layer = packet[TCP]
            flags = tcp_layer.flags
            feature_dict['tcp_flags']['SYN'] += 1 if flags & 0x02 else 0
            feature_dict['tcp_flags']['ACK'] += 1 if flags & 0x10 else 0
            feature_dict['tcp_flags']['FIN'] += 1 if flags & 0x01 else 0
            feature_dict['tcp_flags']['RST'] += 1 if flags & 0x04 else 0

            feature_dict['src_ports'][tcp_layer.sport] += 1
            feature_dict['dst_ports'][tcp_layer.dport] += 1

        # Check for UDP layer and extract ports
        if packet.haslayer(UDP):
            udp_layer = packet[UDP]
            feature_dict['src_ports'][udp_layer.sport] += 1
            feature_dict['dst_ports'][udp_layer.dport] += 1

        # Check for ICMP layer
        if packet.haslayer(ICMP):
            feature_dict['protocols']['ICMP'] += 1

    # Finalize unique IP counts
    feature_dict['src_ip_count'] = len(feature_dict['unique_src_ips'])
    feature_dict['dst_ip_count'] = len(feature_dict['unique_dst_ips'])
    del feature_dict['unique_src_ips']
    del feature_dict['unique_dst_ips']

    # Calculate additional features
    feature_dict['avg_packet_size'] = (
        sum(feature_dict['packet_sizes']) / len(feature_dict['packet_sizes']) if feature_dict['packet_sizes'] else 0
    )
    feature_dict['src_port_entropy'] = calculate_entropy(feature_dict['src_ports'])
    feature_dict['dst_port_entropy'] = calculate_entropy(feature_dict['dst_ports'])
    
    # Calculate protocol percentages
    total_protocols = sum(feature_dict['protocols'].values())
    for proto in feature_dict['protocols']:
        feature_dict[f'protocol_{proto}_percentage'] = (
            feature_dict['protocols'][proto] / total_protocols if total_protocols else 0
        )

    # Clean up features for DataFrame compatibility
    del feature_dict['packet_sizes']  # Removing large raw data lists
    del feature_dict['tcp_flags']     # Can be unpacked if needed

    # Convert feature dictionary to DataFrame
    df = pd.DataFrame([feature_dict])
    
    return df

def extract_features_from_folder(folder_path, label):
    feature_list = []
    for file in os.listdir(folder_path):
        if file.endswith('.pcap'):
            pcap_path = os.path.join(folder_path, file)
            print(f"Processing {pcap_path}")
            df = extract_features_from_file(pcap_path)
            df['label'] = label  # Add label (0 for benign, 1 for DDoS)
            feature_list.append(df)
    return pd.concat(feature_list, ignore_index=True)