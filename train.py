import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from dataset import SoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = .001
ANNOTATION = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\annotation\\hr_bot_synt.json"
AUDIO = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\hr_bot_synt"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050

'''def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader'''

def train_one_epoch(model, data_loder, loss_fn, optimiser, device):
    for input, target in data_loder:
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Loss: {loss.item()}")

def train(model, data_loder, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch: {i+1}")
        train_one_epoch(model, data_loder, loss_fn, optimiser, device)
        print("------------------")
    print("Training is done.")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    usd = SoundDataset(ANNOTATION, AUDIO, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    train_dataloader = DataLoader(usd, BATCH_SIZE)

    cnn = CNNNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnn.pth")
    print("Model trainded and stored at cnn.pth")