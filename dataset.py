from torch.utils.data import Dataset
import torch
import torchaudio
import json
import os

class SoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        lenght_signal = signal.shape[1]
        if lenght_signal < self.num_samples:
            num_missing_samples = self.num_samples - lenght_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        filename = self.annotations[index]["audio_filepath"]
        return os.path.join(self.audio_dir, filename)
        
    def _get_audio_sample_label(self, index):
        return int(self.annotations[index]['label'])
    
if __name__ == "__main__":
    ANNOTATION = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\annotation\\hr_bot_synt.json"
    AUDIO = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\hr_bot_synt"
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 22050

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using: {device}")

    usd = SoundDataset(ANNOTATION, AUDIO, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples")

    signal, label = usd[0]