import numpy as np
import torch 
import torchaudio

from lib_phmm.config import CONFIG


def load_file(filename, model, processor):
    print(f"... {filename}")
    waveform, _ = load_filtered_waveform(filename)
    results     = [result for result in process(waveform, model, processor)]    
    
    embeddings = [r['embeddings'] for r in results]
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
    
    classifications = [classify(r['classifications']) for r in results]
    annotated       = [x for x in zip(classifications, embeddings)]
    
    sequence   = [a[1] for a in annotated if a[0] != 'NOISE']
        
    return sequence, embeddings, classifications


def load_filtered_waveform(filename):
    waveform, sr = torchaudio.load(filename)
    waveform = torchaudio.functional.highpass_biquad(
        waveform, sr, CONFIG['min_frequency_hz']
    )
    waveform = torchaudio.functional.lowpass_biquad(
        waveform, sr, CONFIG['max_frequency_hz']
    )
    return waveform, sr


def process(waveform, model, processor):
    window_size_sample = CONFIG['window_size_samples'] // CONFIG['step_denominator']
    window_size = CONFIG['window_size_samples']
    with torch.no_grad():
        total_len = waveform.shape[1]    
        for i in range(0, total_len, window_size_sample):
            x = waveform[:, i:i + window_size]
            device = model.device
            inputs = processor(
                x.squeeze(),
                sampling_rate=CONFIG['sampling_rate'],
                return_tensors="pt",
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
            )
            mask = np.zeros((1, CONFIG['attention_mask_length']))
            mask[0] = 1
            input_features = inputs['input_features'].reshape((1, CONFIG['mel_bins'], CONFIG['time_frames'])).to(device)
            input_mask = torch.tensor(mask, dtype=torch.float).to(device)
        
            result = model.forward(
                input_features,
                input_mask=input_mask
            )
            yield result


LABEL_MAP = {
    1: 'DOWN',
    2: 'UP',
    3: 'BURST',
    4: 'ECHO',
    0: 'NOISE'
}


def classify(y):
    probs = torch.softmax(y[0, 0, :], dim=0).detach().cpu().numpy()
    confidence = probs.max()
    predicted_class = int(probs.argmax())
    label = LABEL_MAP[predicted_class]
    return label
