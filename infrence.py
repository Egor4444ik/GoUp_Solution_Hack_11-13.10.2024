import torch
from cnn import CNNNetwork
import torchaudio
from dataset import SoundDataset

class_mapping=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22"]
ANNOTATION = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\annotation\\hr_bot_synt.json"
AUDIO = "C:\\MyProjects\\Python\\ML_test\\ESC_DATASET_v1.2\\hr_bot_synt"
SAMPLE_RATE = 16000
NUM_SAMPLES = 22050

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        excepted = class_mapping[target]
    return predicted, excepted



if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    cnn = CNNNetwork()
    state_dict = torch.load("cnn.pth")
    cnn.load_state_dict(state_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = SoundDataset(ANNOTATION, AUDIO, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    for i in range(100):
        input, target = usd[i][0], usd[i][1]
        input.unsqueeze_(0)

        predicted, expected = predict(cnn, input, target, class_mapping)

        print(f"predicted: '{predicted}', expected: '{expected}'")