import torch
import torch.nn.functional as F

def joint_capmeme_loss(logits, sarcasm_target, A_v, A_t, emotion_target, bce_weight=1.0, affect_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, sarcasm_target.float().to(logits.device), reduction="mean")
    if A_v is None or A_t is None or affect_weight == 0:
        return bce_weight * bce, bce, torch.tensor(0.0, device=logits.device)
    emotion_target = emotion_target.to(logits.device).float()
    affect_loss = F.mse_loss(A_v, emotion_target, reduction="mean") + F.mse_loss(A_t, emotion_target, reduction="mean")
    return bce_weight * bce + affect_weight * affect_loss, bce, affect_loss
