import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel

class KGModule(nn.Module):
    def __init__(self, text_dim, kg_dim):
        super().__init__()
        self.fallback = nn.Linear(text_dim, kg_dim)

    def forward(self, text_embedding, kg_embedding=None, kg_valid=None):
        if kg_embedding is not None and kg_valid is not None and (kg_valid > 0).any():
            out = torch.where(kg_valid.unsqueeze(1) > 0.5, kg_embedding, self.fallback(text_embedding))
        else:
            out = self.fallback(text_embedding)
        return out

class CAPMeme(nn.Module):
    def __init__(self, num_emotions, kg_dim=256, fusion_hidden=128, clip_name="openai/clip-vit-base-patch32", text_model_name="bert-base-multilingual-cased", vis_dim=768, text_dim=768):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        self.num_emotions = num_emotions
        self.kg_dim = kg_dim
        self.vis_affect = nn.Linear(vis_dim, num_emotions)
        self.text_affect = nn.Linear(text_dim, num_emotions)
        self.kg_module = KGModule(text_dim, kg_dim)
        self.fusion = nn.Linear(num_emotions + kg_dim, fusion_hidden)
        self.classifier = nn.Linear(fusion_hidden, 1)

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        E_v = vis_out.pooler_output
        if E_v is None:
            E_v = vis_out.last_hidden_state[:, 0]
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        E_t = text_out.last_hidden_state[:, 0]
        A_v = self.vis_affect(E_v)
        A_t = self.text_affect(E_t)
        F_contrast = torch.abs(A_v - A_t)
        E_kg = self.kg_module(E_t, kg_embedding, kg_valid)
        fused = torch.cat([F_contrast, E_kg], dim=-1)
        fused = torch.relu(self.fusion(fused))
        logits = self.classifier(fused).squeeze(-1)
        return logits, A_v, A_t


# ---------------------------------------------------------------------------
# Model registry and baselines (RESEARCH_PLAN_TOP_JOURNAL.md)
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "capmeme", "capmeme_no_kg", "capmeme_no_emotion", "capmeme_concat_fusion",
    "text_only", "image_only", "late_fusion",
]


class TextOnlyMeme(nn.Module):
    def __init__(self, text_model_name="bert-base-multilingual-cased", text_dim=768, hidden=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(text_model_name)
        self.classifier = nn.Sequential(nn.Linear(text_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        E_t = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(E_t).squeeze(-1), None, None


class ImageOnlyMeme(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", vis_dim=768, hidden=128, **kwargs):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(clip_name)
        self.classifier = nn.Sequential(nn.Linear(vis_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        out = self.encoder(pixel_values=pixel_values)
        E_v = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        return self.classifier(E_v).squeeze(-1), None, None


class LateFusionMeme(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch32", text_model_name="bert-base-multilingual-cased", vis_dim=768, text_dim=768, hidden=256):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.fusion = nn.Sequential(nn.Linear(vis_dim + text_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        E_v = vis_out.pooler_output if vis_out.pooler_output is not None else vis_out.last_hidden_state[:, 0]
        E_t = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.fusion(torch.cat([E_v, E_t], dim=-1)).squeeze(-1), None, None


class CAPMemeNoEmotion(nn.Module):
    def __init__(self, num_emotions, kg_dim=256, fusion_hidden=128, **kwargs):
        super().__init__()
        self.core = CAPMeme(num_emotions=num_emotions, kg_dim=kg_dim, fusion_hidden=fusion_hidden, **kwargs)

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        return self.core(pixel_values, input_ids, attention_mask, kg_embedding, kg_valid)


class CAPMemeConcatFusion(nn.Module):
    def __init__(self, num_emotions, kg_dim=256, fusion_hidden=128, clip_name="openai/clip-vit-base-patch32", text_model_name="bert-base-multilingual-cased", vis_dim=768, text_dim=768):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.vis_affect = nn.Linear(vis_dim, num_emotions)
        self.text_affect = nn.Linear(text_dim, num_emotions)
        self.kg_module = KGModule(text_dim, kg_dim)
        self.fusion = nn.Linear(vis_dim + text_dim + kg_dim, fusion_hidden)
        self.classifier = nn.Linear(fusion_hidden, 1)

    def forward(self, pixel_values, input_ids, attention_mask, kg_embedding=None, kg_valid=None):
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        E_v = vis_out.pooler_output if vis_out.pooler_output is not None else vis_out.last_hidden_state[:, 0]
        E_t = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        A_v, A_t = self.vis_affect(E_v), self.text_affect(E_t)
        E_kg = self.kg_module(E_t, kg_embedding, kg_valid)
        fused = torch.relu(self.fusion(torch.cat([E_v, E_t, E_kg], dim=-1)))
        return self.classifier(fused).squeeze(-1), A_v, A_t


def build_model(model_name: str, num_emotions: int, kg_dim: int = 256, fusion_hidden: int = 128, **kwargs) -> nn.Module:
    if model_name in ("capmeme", "capmeme_no_kg"):
        return CAPMeme(num_emotions=num_emotions, kg_dim=kg_dim, fusion_hidden=fusion_hidden, **kwargs)
    if model_name == "capmeme_no_emotion":
        return CAPMemeNoEmotion(num_emotions=num_emotions, kg_dim=kg_dim, fusion_hidden=fusion_hidden, **kwargs)
    if model_name == "capmeme_concat_fusion":
        return CAPMemeConcatFusion(num_emotions=num_emotions, kg_dim=kg_dim, fusion_hidden=fusion_hidden, **kwargs)
    if model_name == "text_only":
        return TextOnlyMeme(**kwargs)
    if model_name == "image_only":
        return ImageOnlyMeme(**kwargs)
    if model_name == "late_fusion":
        return LateFusionMeme(**kwargs)
    raise ValueError(f"Unknown model: {model_name}. Choose from {MODEL_NAMES}")
