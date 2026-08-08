from __future__ import annotations

try:
    import torch
    from torch import nn
    from torchvision import models
except ImportError:  # Permite auditoria e baselines sem stack neural.
    torch = None
    nn = None
    models = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch/torchvision nao instalados. Execute: pip install -r Modelo_novo/requirements.txt")


if nn is not None:
    class Encoder(nn.Module):
        def __init__(self, embedding_dim=64, pretrained=True, freeze_backbone=True, dropout=0.15):
            super().__init__()
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            network = models.resnet18(weights=weights)
            features = network.fc.in_features
            network.fc = nn.Identity()
            self.backbone = network
            self.projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, embedding_dim), nn.LayerNorm(embedding_dim))
            if freeze_backbone:
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

        def forward(self, image):
            return nn.functional.normalize(self.projection(self.backbone(image)), dim=-1)

        def unfreeze_last_block(self):
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True


    class GeometryEncoder(nn.Module):
        def __init__(self,input_dim=128,hidden_dim=128,embedding_dim=64,dropout=.2):
            super().__init__()
            self.network=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,embedding_dim),nn.LayerNorm(embedding_dim))
        def forward(self,values): return nn.functional.normalize(self.network(values),dim=-1)


    class TwoTower(nn.Module):
        """Torres independentes: face e cranio nunca compartilham pesos."""
        def __init__(self, embedding_dim=64, pretrained=True, freeze_backbone=True, dropout=0.15, view_fusion="mean", max_skull_views=3, use_geometry=False, geometry_hidden_dim=128, face_view_fusion="mean", max_face_views=3):
            super().__init__()
            self.face_encoder = Encoder(embedding_dim, pretrained, freeze_backbone, dropout)
            self.skull_encoder = Encoder(embedding_dim, pretrained, freeze_backbone, dropout)
            self.view_fusion = str(view_fusion)
            self.face_view_fusion = str(face_view_fusion)
            self.use_geometry=bool(use_geometry)
            if self.use_geometry:
                self.face_geometry=GeometryEncoder(128,geometry_hidden_dim,embedding_dim,dropout)
                self.skull_geometry=GeometryEncoder(128,geometry_hidden_dim,embedding_dim,dropout)
                self.face_hybrid=nn.Sequential(nn.Linear(embedding_dim*2,embedding_dim),nn.LayerNorm(embedding_dim))
                self.skull_hybrid=nn.Sequential(nn.Linear(embedding_dim*2,embedding_dim),nn.LayerNorm(embedding_dim))
            if self.view_fusion == "attention":
                self.view_position = nn.Parameter(torch.zeros(max_skull_views, embedding_dim))
                nn.init.normal_(self.view_position, std=0.02)
                self.view_attention = nn.Sequential(nn.Linear(embedding_dim, embedding_dim//2), nn.Tanh(), nn.Linear(embedding_dim//2, 1))
            elif self.view_fusion != "mean":
                raise ValueError(f"Fusão de vistas desconhecida: {self.view_fusion}")

            if self.face_view_fusion == "attention":
                self.face_view_position = nn.Parameter(torch.zeros(max_face_views, embedding_dim))
                nn.init.normal_(self.face_view_position, std=0.02)
                self.face_view_attention = nn.Sequential(nn.Linear(embedding_dim, embedding_dim//2), nn.Tanh(), nn.Linear(embedding_dim//2, 1))
            elif self.face_view_fusion != "mean":
                raise ValueError(f"Fusão de vistas faciais desconhecida: {self.face_view_fusion}")

        def fuse_skull_views(self, encoded, batch, views):
            encoded=encoded.reshape(batch,views,-1)
            if self.view_fusion == "attention":
                if views > len(self.view_position): raise ValueError("Número de vistas maior que o configurado")
                enriched=encoded+self.view_position[:views].unsqueeze(0)
                weights=torch.softmax(self.view_attention(enriched).squeeze(-1),dim=1)
                return nn.functional.normalize((encoded*weights.unsqueeze(-1)).sum(dim=1),dim=-1)
            return nn.functional.normalize(encoded.mean(dim=1),dim=-1)

        def fuse_face_views(self, encoded, batch, views):
            encoded=encoded.reshape(batch,views,-1)
            if self.face_view_fusion == "attention":
                if views > len(self.face_view_position): raise ValueError("Número de vistas faciais maior que o configurado")
                enriched=encoded+self.face_view_position[:views].unsqueeze(0)
                weights=torch.softmax(self.face_view_attention(enriched).squeeze(-1),dim=1)
                return nn.functional.normalize((encoded*weights.unsqueeze(-1)).sum(dim=1),dim=-1)
            return nn.functional.normalize(encoded.mean(dim=1),dim=-1)

        def forward(self, face, skull, face_shape=None, skull_shape=None):
            if face.ndim == 5:
                face_batch, face_views, channels, height, width = face.shape
                encoded_face = self.face_encoder(face.reshape(face_batch * face_views, channels, height, width))
                face_embedding = self.fuse_face_views(encoded_face, face_batch, face_views)
            else:
                face_embedding = self.face_encoder(face)
            if skull.ndim == 5:
                batch, views, channels, height, width = skull.shape
                encoded = self.skull_encoder(skull.reshape(batch * views, channels, height, width))
                skull_embedding = self.fuse_skull_views(encoded,batch,views)
            else:
                skull_embedding = self.skull_encoder(skull)
            if self.use_geometry:
                if face_shape is None or skull_shape is None: raise ValueError("O modelo híbrido requer os shapes faciais e cranianos")
                face_geometry=self.face_geometry(face_shape)
                skull_geometry=self.skull_geometry(skull_shape)
                face_embedding=nn.functional.normalize(self.face_hybrid(torch.cat([face_embedding,face_geometry],dim=-1)),dim=-1)
                skull_embedding=nn.functional.normalize(self.skull_hybrid(torch.cat([skull_embedding,skull_geometry],dim=-1)),dim=-1)
            return face_embedding, skull_embedding

        def unfreeze_last_blocks(self):
            self.face_encoder.unfreeze_last_block()
            self.skull_encoder.unfreeze_last_block()
else:
    class TwoTower:  # type: ignore
        def __init__(self, *args, **kwargs):
            require_torch()


def symmetric_infonce(face_embeddings, skull_embeddings, temperature=0.07, label_smoothing=0.0):
    require_torch()
    logits = face_embeddings @ skull_embeddings.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing) + nn.functional.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)) / 2


def symmetric_hard_negative_loss(face_embeddings, skull_embeddings, temperature=0.07, label_smoothing=0.0, margin=0.2, top_k=3, weight=0.5):
    """InfoNCE mais uma penalização sobre os negativos mais parecidos de cada lote."""
    base = symmetric_infonce(face_embeddings, skull_embeddings, temperature, label_smoothing)
    similarity = face_embeddings @ skull_embeddings.T
    batch_size = similarity.shape[0]
    if batch_size < 2 or weight <= 0:
        return base, base.detach().new_zeros(())
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=similarity.device)
    k = min(max(1, int(top_k)), batch_size - 1)

    def directional(matrix):
        positives = matrix.diagonal().unsqueeze(1)
        negatives = matrix.masked_fill(diagonal, float("-inf")).topk(k, dim=1).values
        return nn.functional.relu(float(margin) + negatives - positives).mean()

    hard = (directional(similarity) + directional(similarity.T)) / 2
    return base + float(weight) * hard, hard
