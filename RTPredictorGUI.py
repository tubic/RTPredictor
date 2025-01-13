import tkinter as tk
from tkinter import filedialog, ttk
import time
import threading
from torch import nn
import torch
import numpy as np
from Bio import SeqIO
from collections import Counter, defaultdict
from itertools import product
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

class RTPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RT Predictor")
        self.root.geometry("500x450")
        self.root.resizable(False, False)

        title_label = tk.Label(root, text="RT-Predictor", font=("Arial", 16, "bold"))
        title_label.pack(pady=20)

        # Select device: CPU or GPU
        self.device_label = tk.Label(root, text="Device:", font=("Arial", 12))
        self.device_label.pack(anchor="w", padx=20)
        self.device_option = ttk.Combobox(root, values=["CPU", "GPU"], state="readonly", font=("Arial", 12))
        self.device_option.set("CPU") 
        self.device_option.pack(pady=5, padx=20, fill="x")

        # Select a fasta file
        self.fasta_label = tk.Label(root, text="Select a fasta file:", font=("Arial", 12))
        self.fasta_label.pack(anchor="w", padx=20)
        self.fasta_path = tk.Entry(root, font=("Arial", 12), state="readonly")
        self.fasta_path.pack(pady=5, padx=20, fill="x")
        self.fasta_button = tk.Button(root, text="Select", command=self.select_fasta_file, font=("Arial", 12))
        self.fasta_button.pack(pady=5, padx=20)

        # Select a output fold
        self.output_label = tk.Label(root, text="Select a output fold：", font=("Arial", 12))
        self.output_label.pack(anchor="w", padx=20)
        self.output_path = tk.Entry(root, font=("Arial", 12), state="readonly")
        self.output_path.pack(pady=5, padx=20, fill="x")
        self.output_button = tk.Button(root, text="Select", command=self.select_output_path, font=("Arial", 12))
        self.output_button.pack(pady=5, padx=20)

        # Start working
        self.submit_button = tk.Button(root, text="Start", command=self.start_process, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white")
        self.submit_button.pack(pady=30)

    def select_fasta_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Fasta Files", "*.fasta;*.fa")])
        if file_path:
            self.fasta_path.config(state="normal")
            self.fasta_path.delete(0, tk.END)
            self.fasta_path.insert(0, file_path)
            self.fasta_path.config(state="readonly")

    def select_output_path(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_path.config(state="normal")
            self.output_path.delete(0, tk.END)
            self.output_path.insert(0, folder_path)
            self.output_path.config(state="readonly")

    def start_process(self):
        device = self.device_option.get()
        fasta_file = self.fasta_path.get()
        output_folder = self.output_path.get()
        if not fasta_file or not output_folder:
            print("Please select the complete file and output path!")
            return None
        print(f"Device:{device}")
        print(f"Fasta path: {fasta_file}")
        print(f"Output fold: {output_folder}")
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Processing...")
        self.progress_window.geometry("400x200")
        self.progress_window.resizable(False, False)
        self.progress_label = tk.Label(self.progress_window, text="Processing...", font=("Arial", 12))
        self.progress_label.pack(pady=10)
        self.progress = ttk.Progressbar(self.progress_window, length=300, mode="determinate", maximum=100)
        self.progress.pack(pady=20)
        self.progress['value'] = 0
        threading.Thread(target=self.simulate_processing).start()

    def simulate_processing(self):
        sequences = list(SeqIO.parse(self.fasta_path.get(), 'fasta'))
        sequence_nums = len(sequences)
        device = torch.device("cuda:0" if torch.cuda.is_available() and self.device_option.get() == 'GPU' else "cpu")
        model = RTModel().to(device)
        model.load_state_dict(torch.load(f'Position_Model_4.pkl', map_location=device, weights_only=True))
        predict_result_file = open(f'{self.output_path.get()}/predict_result.txt', 'w', encoding='utf-8')
        for seq_id, seq in enumerate(sequences):
            predict(seq, model, device, predict_result_file)
            self.progress['value'] = ((seq_id + 1) / sequence_nums) * 100
            self.progress_window.update_idletasks()  
        self.progress_label.config(text="Finished.")
        time.sleep(1)
        predict_result_file.close()
        self.progress_window.destroy() 


root = tk.Tk()
app = RTPredictorGUI(root)
root.mainloop()
