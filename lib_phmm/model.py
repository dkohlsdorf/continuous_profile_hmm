import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import WhisperModel, WhisperProcessor
from peft import get_peft_model, LoraConfig
from lib_phmm.config import CONFIG, EMBEDDING_MODELS


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



class WDPGemmaEmbModel(nn.Module):
    """
    EmbeddingGemma2 audio tower + LoRA with the embedding/classifier heads
    from wdp-heads.pt (see analysis_notebooks/EmbeddingGemma2Eval.ipynb).
    Takes raw (B, win) sample windows instead of precomputed features: the
    feature extractor is part of the model here.
    """
    kind = 'gemma'

    def __init__(self, tower, feature_extractor, config, p_drop=0.1):
        super().__init__()
        self.encoder = tower
        self.feature_extractor = feature_extractor
        self.sr = config["sr"]
        self.win = config["win"]
        self.tower_takes_mask = len(inspect.signature(tower.forward).parameters) > 1
        with torch.no_grad():                                   # measure tower width once
            d = self.run_tower(torch.zeros(1, self.win)).shape[-1]
        self.head_embedding = nn.Linear(d, config["proj_dim"])
        self.classifier = nn.Sequential(nn.Dropout(p_drop), nn.Linear(config["proj_dim"], config["num_classes"]))

    @property
    def device(self):
        return next(self.parameters()).device

    def run_tower(self, x):
        dev = next(self.encoder.parameters()).device
        f = self.feature_extractor(list(x.detach().cpu().numpy()), sampling_rate=self.sr, return_tensors="pt")
        feats = f["input_features"].to(dev)
        out = (self.encoder(feats, f["input_features_mask"].to(dev))
               if self.tower_takes_mask else self.encoder(feats))
        return out[0] if isinstance(out, tuple) else out.last_hidden_state

    def forward(self, x):                                       # (B, win) -> (B, proj_dim), (B, n_classes)
        z = self.head_embedding(self.run_tower(x).mean(1).float())
        return {"classifications": self.classifier(z), "embeddings": F.normalize(z, dim=-1)}


def gemma_model(model_dir):
    from transformers import AutoConfig, AutoModel, AutoFeatureExtractor
    from peft import PeftModel
    from lib_phmm.signals import LABEL_MAP

    ck = torch.load(f"{model_dir}/wdp-heads.pt", map_location="cpu")
    config = ck["config"]
    # the heads were trained with WSTL_UP/WSTL_DOWN; the pipeline calls them UP/DOWN
    assert {k: v.replace("WSTL_", "") for k, v in config["label_map"].items()} == LABEL_MAP, config["label_map"]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading EmbeddingGemma2 model on device: {device}")
    print(f"Loading base model {config['base_model']} + adapter/heads from: {model_dir}")
    cfg = AutoConfig.from_pretrained(config["base_model"])
    cfg.vision_config = None                                    # audio tower only
    base = AutoModel.from_pretrained(config["base_model"], config=cfg, torch_dtype=torch.float32)
    tower = next(m for m in base.modules() if type(m).__name__.endswith("AudioModel"))
    del base
    tower = PeftModel.from_pretrained(tower, f"{model_dir}/audio-lora")
    feature_extractor = AutoFeatureExtractor.from_pretrained(config["base_model"])

    m = WDPGemmaEmbModel(tower, feature_extractor, config)
    m.head_embedding.load_state_dict(ck["head_embedding"])
    m.classifier.load_state_dict(ck["classifier"])
    return m.to(device).eval()


def load_embedding_model(name):
    """(model, processor) for an EMBEDDING_MODELS key; processor is None for
    Gemma models, which carry their own feature extractor."""
    print(f"Embedding model: {name}")
    if name == "whisper":
        return whisper_model_v2(model_file=EMBEDDING_MODELS[name]), whisper_processor()
    return gemma_model(EMBEDDING_MODELS[name]), None
