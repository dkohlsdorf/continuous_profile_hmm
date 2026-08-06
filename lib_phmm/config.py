CONFIG = {
    "embedding_dimension": 128,
    "num_classes": 5,
    "triplet_margin": 1.0,
    "triplet_weight": 0.5,
    "classification_weight": 1.0,
    "base_model_id": "openai/whisper-small",
    "checkpoint_path": "model/wdp_whisper_emb_model_L2.pt",
    "min_frequency_hz": 5000,
    "max_frequency_hz": 7900,
    "window_size_samples": 5120,
    "step_denominator": 2,
    "sampling_rate": 16000,
    "attention_mask_length": 1501,
    "mel_bins": 80,
    "time_frames": 3000

}
