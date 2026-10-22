import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MeanPooling(nn.Module):
    """Mean Pooling: Tính trung bình cộng các token vectors dựa trên attention mask."""
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask


class MultiSampleClassifier(nn.Module):
    """
    Model Classifier tối ưu cho Hackathon:
    - Backbone Transformer (DistilBERT, DeBERTa, XLM-RoBERTa, PhoBERT...)
    - Mean Pooling (nắm bắt ngữ cảnh toàn câu tốt hơn [CLS] token đơn thuần)
    - Multi-Sample Dropout (5 nhánh dropout song song chống overfit và tăng tốc hội tụ)
    """
    def __init__(
        self,
        model_name_or_path: str,
        num_classes: int = 3,
        dropout_rates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.config.update({"output_hidden_states": True})
        self.backbone = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        hidden_size = getattr(self.config, "hidden_size", None) or getattr(
            self.config, "d_model", 768
        )

        self.pooler = MeanPooling()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(hidden_size, num_classes)

        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = getattr(self.config, "initializer_range", 0.02)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, return_dict: bool = False):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        feature = self.pooler(outputs.last_hidden_state, attention_mask)

        # Multi-sample dropout: Cho đi qua 5 nhánh dropout và tính trung bình logits
        logits = torch.mean(
            torch.stack([self.fc(drop(feature)) for drop in self.dropouts], dim=0),
            dim=0,
        )
        return (logits,)
