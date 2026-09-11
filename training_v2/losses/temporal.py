from torch import Tensor

from .rankic import DifferentiableRankICLoss


class TemporalRankICLoss(DifferentiableRankICLoss):
    """RankIC plus the report's bounded linear reward for signal persistence."""

    requires_ordered_batches = True
    requires_epoch_update = True

    def __init__(self, turnover_rate: float = 0.1, temperature: float = 0.01, method: str = "sigmoid"):
        super().__init__(temperature, method)
        self.turnover_rate = float(turnover_rate)

    def forward(self, preds: Tensor, labels: Tensor, context=None) -> Tensor:
        rank_ic = self.correlation(preds, labels)
        if not context or context.get("previous_preds") is None:
            return -rank_ic
        persistence = self.correlation(
            preds.reshape(-1)[context["current_indices"]],
            context["previous_preds"],
        )
        return -(rank_ic + self.turnover_rate * persistence)
