import torch

def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean R² score across output dimensions.
    y_true, y_pred: shape [B, D]
    """
    y_true_mean = y_true.mean(dim=0)
    ss_total = ((y_true - y_true_mean) ** 2).sum(dim=0)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_total + 1e-8)
    return r2.mean()

def pearson_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean Pearson correlation across output dimensions.
    y_true, y_pred: shape [B, D]
    """
    y_true_centered = y_true - y_true.mean(dim=0)
    y_pred_centered = y_pred - y_pred.mean(dim=0)

    numerator = (y_true_centered * y_pred_centered).mean(dim=0)
    denominator = y_true_centered.std(dim=0) * y_pred_centered.std(dim=0)

    corr = numerator / (denominator + 1e-8)
    return corr.mean()

def spearman_corr(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean Spearman correlation across output dimensions.
    Approximates ranks using argsort twice.
    """
    def rank(data: torch.Tensor) -> torch.Tensor:
        """
        Approximate ranks. Returns float tensor of same shape.
        """
        ranks = torch.zeros_like(data, dtype=torch.float)
        for d in range(data.shape[1]):
            _, indices = torch.sort(data[:, d])
            ranks[indices, d] = torch.arange(len(data), dtype=torch.float, device=data.device)
        return ranks

    y_true_ranked = rank(y_true)
    y_pred_ranked = rank(y_pred)
    return pearson_corr(y_true_ranked, y_pred_ranked)
