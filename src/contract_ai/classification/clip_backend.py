from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open_clip
import torch
from PIL import Image


@dataclass
class ClipBackend:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

    @torch.inference_mode()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)[0]

    @torch.inference_mode()
    def zero_shot_scores(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        prompts = [f"a product photo of a {label}" for label in labels]
        txt = self.tokenizer(prompts).to(self.device)

        image_f = self.model.encode_image(image_tensor)
        text_f = self.model.encode_text(txt)
        image_f /= image_f.norm(dim=-1, keepdim=True)
        text_f /= text_f.norm(dim=-1, keepdim=True)

        sims = (100.0 * image_f @ text_f.T).softmax(dim=-1).cpu().numpy()[0]
        return {label: float(score) for label, score in zip(labels, sims)}
