import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoFeatureExtractor, WhisperModel, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType
from lib_phmm.config import CONFIG


class WDPWhisperEmbModelV2(nn.Module):
    kind = 'whisper'

    def __init__(self, lora_encoder, output_dim_embedding=None, n_classes=None,
                 win_frames=16, step_frames=8):
        super().__init__()
        self.encoder = lora_encoder
        self.output_dim_embedding = output_dim_embedding if output_dim_embedding is not None else CONFIG['embedding_dimension']
        self.n_classes = n_classes if n_classes is not None else CONFIG['num_classes']
        self.win_frames = win_frames    # 5120 samples / 320 per frame
        self.step_frames = step_frames  # 2560 samples / 320 per frame

        self.head_embedding = nn.Linear(self.encoder.config.d_model, self.output_dim_embedding)
        self.classifier = nn.Linear(self.encoder.config.d_model, self.n_classes)

    @property
    def device(self):
        """Return the device the model is on."""
        return next(self.parameters()).device

    def embed(self, input_features, input_mask):
        out = self.encoder(input_features=input_features)
        emb = out.last_hidden_state
        seq_len = emb.shape[1]

        pooled_mask = F.adaptive_avg_pool2d(input_mask.unsqueeze(1), (1, seq_len)).squeeze(1).squeeze(1)
        mask_expanded = pooled_mask.unsqueeze(-1).expand_as(emb)
        masked_emb = emb * mask_expanded
        sum_emb = masked_emb.sum(dim=1)
        seq_lengths = pooled_mask.sum(dim=1, keepdim=True).clamp(min=1)
        emb_pooled = sum_emb / seq_lengths
        embeddings = self.head_embedding(emb_pooled)
        return emb, embeddings, pooled_mask

    def windowed_classify(self, emb, pooled_mask, target_len):
        x = emb.transpose(1, 2)                                  # [B, D, seq_len]
        m = pooled_mask.unsqueeze(1).to(x.dtype)                # [B, 1, seq_len]
        k, s = self.win_frames, self.step_frames
        num = F.avg_pool1d(x * m, k, s, ceil_mode=True)
        den = F.avg_pool1d(m,     k, s, ceil_mode=True)
        wavg = (num / den.clamp(min=1e-6)).transpose(1, 2)
        logits = self.classifier(wavg)

        if target_len is not None and logits.shape[1] != target_len:   # pad to the label grid
            if logits.shape[1] < target_len:
                logits = F.pad(logits, (0, 0, 0, target_len - logits.shape[1]))
            else:
                logits = logits[:, :target_len]
        return logits

    def forward(self, input_features, sequence_labels=None, sequence_positive=None,
                sequence_negative=None, sequence_mask=None, input_mask=None):
        emb, embeddings, pooled_mask = self.embed(input_features, input_mask)

        target_len = sequence_labels.shape[1] if sequence_labels is not None else None
        classifications = self.windowed_classify(emb, pooled_mask, target_len)

        if sequence_labels is not None:
            active_positions = sequence_mask.view(-1) == 1
            active_logits = classifications.view(-1, self.n_classes)[active_positions]
            active_labels = sequence_labels.view(-1)[active_positions]
            classification_loss = F.cross_entropy(active_logits, active_labels)

            triplet_loss = 0
            if sequence_positive is not None and sequence_negative is not None:
                _, pos_embeddings, _ = self.embed(sequence_positive, input_mask)
                _, neg_embeddings, _ = self.embed(sequence_negative, input_mask)
                triplet_loss = F.triplet_margin_loss(
                    embeddings,
                    pos_embeddings,
                    neg_embeddings,
                    margin=CONFIG['triplet_margin']
                )

            total_loss = CONFIG['triplet_weight'] * triplet_loss + CONFIG['classification_weight'] * classification_loss

            return {
                "loss": total_loss,
                "embeddings": embeddings,
                "classifications": classifications,
            }
        else:
            return {
                "embeddings": embeddings,
                "classifications": classifications
            }


def whisper_processor(model_size=None, model_id=None):
    if model_id is None:
        model_id = CONFIG['base_model_id']
    print(f"Loading Whisper processor from: {model_id}")
    return WhisperProcessor.from_pretrained(model_id)


def whisper_model_v2(model_size=None, model_file=None, cuda=None):
    if model_file is None:
        model_file = CONFIG['checkpoint_path']
    if cuda is None:
        cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Loading Whisper V2 model on device: {device}")
    print(f"Loading checkpoint from: {model_file}")
    m = WDPWhisperEmbModelV2(lora_encoder(model_size))
    m.load_state_dict(torch.load(model_file, map_location=device))
    m.to(device)
    return m


def lora_config():
    return LoraConfig(
        inference_mode=False,
        r=16,                        # 8 -> 16 (or 32 if VRAM allows)
        lora_alpha=64,               # scale up with r (try 64–128)
        lora_dropout=0.0,           # 0.1 -> 0.05 (or 0.0)
        bias="none",
        target_modules=[             # cover attn + MLP
            "q_proj", "k_proj", "v_proj", "out_proj",
            "fc1", "fc2"
        ],
    )


def base_model(model_size=None, model_id=None):
    if model_id is None:
        model_id = CONFIG['base_model_id']
    print(f"Loading base model: {model_id}")
    return WhisperModel.from_pretrained(model_id)


def lora_encoder(model_size=None):
    return get_peft_model(base_model(model_size).encoder, lora_config())


def raw(path):
    try:
        fs, x = read(path)
        if len(x.shape) > 1:
            return fs, x[:, 0]
        return fs, x
    except:
        print("Could not read file: {}".format(path))
        return 0, np.zeros(0)
