from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from v3.models.registry import register_model


PRIME_Config = {
    "name": "prime",
    "params": {
        "block_dims": {"bull": 16, "bear": 10, "friction": 9, "macro": 9},
        "hidden_size": 32,
        "energy_size": 16,
        "dropout": 0.3,
        "sequence_pool": "last",
        "input_noise_std": 0.03,
        "heat_penalty": 0.5,
    },
}


class FeatureEnergyEncoder(nn.Module):
    """Feature-direction-aware encoder used by each PRIME force field."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super().__init__()
        self.direction = nn.Parameter(torch.zeros(input_size))
        self.gate = nn.Parameter(torch.zeros(input_size))
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
        )
        self.energy = nn.Sequential(nn.Linear(output_size, output_size), nn.GELU(), nn.Linear(output_size, 1))

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        directed = inputs * torch.tanh(self.direction) * (2.0 * torch.sigmoid(self.gate))
        hidden = self.network(directed)
        return F.softplus(self.energy(hidden)), hidden

    def direction_regularization(self) -> Tensor:
        return -torch.tanh(self.direction).abs().mean()


class TripleMacroModulation(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU()
        )
        self.head = nn.Linear(hidden_size, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = torch.sigmoid(self.head(self.shared(inputs)))
        alpha = 0.5 + 1.5 * raw[:, 0:1]
        beta = 0.5 + 1.5 * raw[:, 1:2]
        gamma = raw[:, 2:3]
        return alpha, beta, gamma


@register_model("prime", config_class=PRIME_Config)
class PrimeEnergyModel(nn.Module):
    """PRIME energy game adapted to v3 feature-block dictionaries.

    The public prediction is a score (higher is better), while the decomposed
    energy fields remain available to PRIME-aware losses through model_output.
    """

    required_blocks = ("bull", "bear", "friction", "macro")

    def __init__(
        self,
        block_dims: Mapping[str, int],
        hidden_size: int = 32,
        energy_size: int = 16,
        dropout: float = 0.3,
        sequence_pool: str = "last",
        input_noise_std: float = 0.03,
        heat_penalty: float = 0.5,
    ):
        super().__init__()
        missing = set(self.required_blocks) - set(block_dims)
        if missing:
            raise ValueError(f"PRIME block_dims is missing: {sorted(missing)}")
        if sequence_pool not in {"last", "mean"}:
            raise ValueError("sequence_pool must be 'last' or 'mean'")
        self.block_dims = {name: int(block_dims[name]) for name in self.required_blocks}
        self.sequence_pool = sequence_pool
        self.input_noise_std = float(input_noise_std)
        self.heat_penalty = float(heat_penalty)

        self.bull_encoder = FeatureEnergyEncoder(self.block_dims["bull"], hidden_size, energy_size, dropout)
        self.bear_encoder = FeatureEnergyEncoder(self.block_dims["bear"], hidden_size, energy_size, dropout)
        self.heat_encoder = FeatureEnergyEncoder(self.block_dims["friction"], hidden_size, energy_size, dropout)
        self.macro_encoder = nn.Sequential(
            nn.LayerNorm(self.block_dims["macro"]),
            nn.Linear(self.block_dims["macro"], hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.macro_modulation = TripleMacroModulation(hidden_size, hidden_size)

    def _block(self, features: Mapping[str, Tensor], name: str) -> Tensor:
        if name not in features:
            raise KeyError(f"PRIME requires feature block {name!r}; received {sorted(features)}")
        value = features[name]
        if value.ndim == 3:
            value = value[:, -1] if self.sequence_pool == "last" else value.mean(dim=1)
        if value.ndim != 2 or value.shape[-1] != self.block_dims[name]:
            raise ValueError(
                f"PRIME block {name!r} expects [stocks, *, {self.block_dims[name]}], "
                f"got {tuple(value.shape)}"
            )
        if self.training and self.input_noise_std > 0:
            value = value + torch.randn_like(value) * self.input_noise_std
        return value

    @staticmethod
    def _normalize_energy(energy: Tensor) -> Tensor:
        centered = (energy - energy.mean()) / energy.std(unbiased=False).clamp_min(1e-6)
        return 0.1 + 0.8 * torch.sigmoid(centered * 1.5)

    def forward(self, features: Mapping[str, Tensor]) -> dict[str, Tensor | dict[str, Tensor]]:
        bull = self._block(features, "bull")
        bear = self._block(features, "bear")
        heat = self._block(features, "friction")
        macro = self._block(features, "macro")

        e_bull, h_bull = self.bull_encoder(bull)
        e_bear, h_bear = self.bear_encoder(bear)
        e_heat, h_heat = self.heat_encoder(heat)
        e_bull = self._normalize_energy(e_bull)
        e_bear = self._normalize_energy(e_bear)
        e_heat = self._normalize_energy(e_heat)

        macro_hidden = self.macro_encoder(macro)
        alpha, beta, gamma = self.macro_modulation(macro_hidden)
        bull_force = alpha * e_bull
        bear_force = beta * e_bear
        heat_force = self.heat_penalty * gamma * e_heat
        energy = bear_force - bull_force + heat_force
        score = -energy.squeeze(-1)
        return {
            "preds": score,
            "direction_regularization": self.direction_regularization().reshape(1),
            "components": {
                "energy": energy.squeeze(-1),
                "E_bull": bull_force.squeeze(-1),
                "E_bear": bear_force.squeeze(-1),
                "E_heat": e_heat.squeeze(-1),
                "E_heat_penalty": heat_force.squeeze(-1),
                "alpha_market": alpha.squeeze(-1),
                "beta_risk": beta.squeeze(-1),
                "gamma_heat": gamma.squeeze(-1),
                "h_bull": h_bull,
                "h_bear": h_bear,
                "h_heat": h_heat,
            },
        }

    def direction_regularization(self) -> Tensor:
        return sum(
            encoder.direction_regularization()
            for encoder in (self.bull_encoder, self.bear_encoder, self.heat_encoder)
        )
