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

# --embedding-model choices: the fine-tuned Whisper checkpoint (the original
# pipeline) or one of the EmbeddingGemma2 audio-tower models (a directory
# holding audio-lora/ + wdp-heads.pt). Paths are relative to cwd, like
# CONFIG['checkpoint_path'].
EMBEDDING_MODELS = {
    "whisper": CONFIG["checkpoint_path"],
    "gemma_noiseaug": "model/wdp-embeddinggemma2-decoder_extended_balanced_noiseaug",
    "gemma_noiseaug_hr": "model/wdp-embeddinggemma2-decoder_extended_balanced_noiseaug_hr",
}
DEFAULT_EMBEDDING_MODEL = "whisper"
