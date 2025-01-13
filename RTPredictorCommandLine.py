import argparse
import time
import torch
import numpy as np
from Bio import SeqIO
from collections import Counter, defaultdict
from itertools import product
from torch import nn

torch.set_num_threads(1)

INPUT_CHANNELS = 6
INPUT_DIM = np.power(4, 3) * INPUT_CHANNELS
EMBEDDING_DIM = 1000
NUM_HEADS = 10
HIDDEN_DIM = 1024
FEATURE_SIZE = 100
CLASSES = 4

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * 2, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ELU(),
            nn.Conv1d(out_channels * 2, out_channels * 3, kernel_size=1),
            nn.ELU(),
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ELU(),
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = out + identity
        return out

class ResNet1D(nn.Module):
    def __init__(self, classes):
        super(ResNet1D, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, INPUT_CHANNELS, kernel_size=7, stride=3, padding=3, bias=False),
            nn.ELU(),
        )
        self.layers, self.output_features = self.make_layers(10)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.output_features),
            nn.Linear(self.output_features, classes),
            nn.Sigmoid(),
        )
        print(f'Model building finished. After ResNet feature size:{self.output_features}')

    def make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.ELU(),
            )
        layers = []
        layers.append(BasicBlock1D(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def make_layers(self, layers):
        net_list = []
        start_channel = INPUT_CHANNELS
        end_channel = start_channel + 100
        for layer in range(layers):           
            if layer == 0:
                net_list.append(self.make_layer(start_channel, end_channel, 3, stride=1))
            else:
                net_list.append(self.make_layer(start_channel, end_channel, 3, stride=2))
            start_channel = end_channel
            end_channel = end_channel + 100
        return nn.Sequential(*net_list), start_channel

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.global_avg_pool(x) * self.global_max_pool(x)
        x = torch.flatten(x, 1)   
        x = self.fc(x)           
        return x

class RTModel(nn.Module):
    def __init__(self):
        super(RTModel, self).__init__()
        self.res_1 = ResNet1D(classes=CLASSES)
    
    def forward(self, x):
        x = self.res_1(x)
        return x

def position_and_freq(sequence, nums):
    seq_len = len(sequence)
    base_pair = ['A', 'C', 'G', 'T']
    for _ in range(nums - 1):
        base_pair = [f'{i}{j}' for i, j in product(base_pair, 'ACGT')]
    seq_array = np.array([sequence[i:i + nums] for i in range(seq_len - nums + 1)])
    seq_counter = Counter(seq_array)
    positions = np.arange(1, seq_len - nums + 2) / seq_len
    encoded_list = defaultdict(list)
    for idx, seq in enumerate(seq_array):
        encoded_list[seq].append(positions[idx])
    encoded_seq = []
    for base in base_pair:
        freq = seq_counter.get(base, 0) / len(seq_array)
        position_list = encoded_list.get(base, [])
        encoded_seq.append([freq] + position_list)
    return encoded_seq

def interval_mean_std(encoded_list):
    if len(encoded_list) > 10:
        return [np.mean(encoded_list), np.std([i-j for i, j in zip(encoded_list[1:], encoded_list)]), np.max([i-j for i, j in zip(encoded_list[1:], encoded_list)]),  np.mean([i-j for i, j in zip(encoded_list[1:], encoded_list)]), np.min([i-j for i, j in zip(encoded_list[1:], encoded_list)])]
    else:
        return [0] * 5

def encode_sequence(sequence):
    mono_list = []
    encoded_lists = position_and_freq(sequence, 3)
    for encoded_list in encoded_lists:
        encode_seq = [len(encoded_list) / len(sequence) - 2] + interval_mean_std(encoded_list)
        mono_list.extend(encode_seq)
    return mono_list

def predict(seq, model, device, predict_result_file):
    label_dict = {
        2:'DTZ',
        4:'LRD',
        1:'ERD',
        3:'UTZ',
    }
    seq_seq = str(seq.seq).upper()
    if 'N' not in seq_seq:
        try:
            data = encode_sequence(seq_seq)
        except:
            data = None
        if data is not None:
            data = torch.Tensor(data).to(device)
            data = torch.reshape(data, (1, 1, INPUT_DIM))
            outputs = model(data)
            predict_result = torch.argmax(outputs[0]).item()
            predict_result_file.write(f'{seq.id} {label_dict[predict_result + 1]}\n')
    else:
        predict_result_file.write(f'{seq.id} -1\n')

def start_process(device, fasta_file, output_folder):
    sequences = list(SeqIO.parse(fasta_file, 'fasta'))
    sequence_nums = len(sequences)
    device = torch.device("cuda:0" if torch.cuda.is_available() and device == 'GPU' else "cpu")
    model = RTModel().to(device)
    model.load_state_dict(torch.load(f'Position_Model_4.pkl', map_location=device, weights_only=True))
    with open(f'{output_folder}/predict_result.txt', 'w', encoding='utf-8') as predict_result_file:
        for seq_id, seq in enumerate(sequences):
            predict(seq, model, device, predict_result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-Predictor")
    parser.add_argument('-d', '--device', choices=['CPU', 'GPU'], default='CPU', help="Device to use (CPU or GPU)")
    parser.add_argument('-f', '--fasta', required=True, help="Path to the fasta file")
    parser.add_argument('-o', '--output', required=True, help="Output directory")

    args = parser.parse_args()
    start_process(args.device, args.fasta, args.output)
