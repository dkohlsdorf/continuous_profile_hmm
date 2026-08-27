import numpy as np
import torch 
import torchaudio

from lib_phmm.config import CONFIG


def bounded_regions(annotated, min_size=10):
    recording = False
    current_classifications = []
    current_embeddings = []
    regions = []
    for classification, embedding in annotated:
        if recording and classification == 'NOISE':
            recording = False
            if len(current_embeddings) >= min_size:
                regions.append((current_classifications, current_embeddings))
            current_classifications = []
            current_embeddings = []
        if classification != 'NOISE':
            recording = True
            current_classifications.append(classification)
            current_embeddings.append(embedding)
    if len(current_embeddings) >= min_size:
        regions.append((current_classifications, current_embeddings))
    return regions


def load_file(filename, model, processor, motif_mode=False):
    print(f"... {filename}")
    waveform, _ = load_filtered_waveform(filename)
    results     = [result for result in process(waveform, model, processor)]    
    
    embeddings = [r['embeddings'] for r in results]
    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[2]))
    
    classifications = [classify(r['classifications']) for r in results]
    annotated       = [x for x in zip(classifications, embeddings)]

    if motif_mode:
        sequences = bounded_regions(annotated)
    else:
        non_noise = [a for a in annotated if a[0] != 'NOISE']
        sequences = [([a[0] for a in non_noise], [a[1] for a in non_noise])]
    return sequences, embeddings, classifications


def load_filtered_waveform(filename):
    waveform, sr = torchaudio.load(filename)
    if waveform.shape[0] > 1:
        # process()'s x.squeeze() only drops size-1 dims, so a multi-channel
        # waveform reaches the Whisper feature extractor as 2D instead of
        # the 1D it expects, breaking its internal padding. Every file this
        # pipeline was built/trained against happens to be mono already, so
        # this is a no-op there -- only multi-channel input hits it.
        waveform = waveform[:1]
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
    2: 'DOWN',
    1: 'UP',
    4: 'BURST',
    3: 'ECHO',
    0: 'NOISE'
}


def classify(y):
    probs = torch.softmax(y[0, 0, :], dim=0).detach().cpu().numpy()
    confidence = probs.max()
    predicted_class = int(probs.argmax())
    label = LABEL_MAP[predicted_class]
    return label
