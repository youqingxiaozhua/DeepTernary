from typing import Optional, Sequence, List, Dict

import torch
from torch import nn, Tensor
from mmengine.evaluator import BaseMetric
from mmpretrain.registry import MODELS, METRICS



class RMSD(nn.Module):
    def __init__(self) -> None:
        super(RMSD, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor], reduction='mean', **kwargs) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        if reduction == 'mean':
            return torch.tensor(rmsds).mean()
        elif reduction == 'none':
            return rmsds
        else:
            raise ValueError(f'Unknown reduction {reduction}')


class MSE(nn.Module):

    def forward(self, x, y):
        scores = []
        for i, j in zip(x, y):
            scores.append(nn.MSELoss()(i, j))
        return torch.tensor(scores).mean()


class RMSDfraction(nn.Module):
    def __init__(self, distance) -> None:
        super(RMSDfraction, self).__init__()
        self.distance = distance

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor], **kwargs) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        count = torch.tensor(rmsds) < self.distance
        return 100 * count.sum() / len(count)



class KabschRMSD(nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor], **kwargs) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()



@METRICS.register_module()
class EquiBindMetric(BaseMetric):
    r"""
        >>> # ------------------- Use with Evalutor -------------------
        >>> from mmpretrain.structures import DataSample
        >>> from mmengine.evaluator import Evaluator
        >>> data_samples = [
        ...     DataSample().set_gt_label(0).set_pred_score(torch.rand(10))
        ...     for i in range(1000)
        ... ]
        >>> evaluator = Evaluator(metrics=Accuracy(topk=(1, 5)))
        >>> evaluator.process(data_samples)
        >>> evaluator.evaluate(1000)
        {
            'accuracy/top1': 9.300000190734863,
            'accuracy/top5': 51.20000076293945
        }
    """
    default_prefix: Optional[str] = 'bind'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)


    def process(self, data_batch: Dict[str, list], data_samples: Sequence[dict]):
        """Process one batch of data samples.

        The processed results should be stored in ``self.results``, which will
        be used to computed the metrics when all batches have been processed.

        Args:
            data_batch: A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for i, data_sample in enumerate(data_samples):
            result = {
                'ligs_coords': data_batch['lig_coords'][i],
                'ligs_coords_pred': data_sample['ligs_coords_pred'],
            }
            self.results.append(result)

    def compute_metrics(self, results: List[dict]):
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        concat_results = dict()
        for key in results[0].keys():
            concat_results[key] = [res[key] for res in results]  # does not need to cat for a list of bool
        metrics = self.calculate(**concat_results, geom_losses1=0.) # Hardcode

        return metrics

    def calculate(
        self,
        **kwargs
    ) -> dict:
        metrics = dict()
        metrics['rmsd_less_than_2'] = RMSDfraction(2)(**kwargs)
        metrics['rmsd_less_than_5'] = RMSDfraction(5)(**kwargs)
        metrics['mean_rmsd'] = RMSD()(**kwargs)
        metrics['kabsch_rmsd'] = KabschRMSD()(**kwargs)
        return metrics


def MAE(pred, gt):
    return torch.abs(pred - gt).mean()


@METRICS.register_module()
class TripletBindMetric(EquiBindMetric):
    """
    add RMSD of rec2
    """
    def process(self, data_batch: Dict[str, list], data_samples: Sequence[dict]):
        for i, data_sample in enumerate(data_samples):
            rec2_coords = data_batch['rec2_coords'][i]
            rotation_2 = data_sample['rotation_2']
            translation_2 = data_sample['translation_2']
            rec2_coods_input = data_batch['rec2_coords_input'][i]
            rec2_coords_pred = (rotation_2 @ rec2_coods_input.t()).t() + translation_2
            result = {
                'ligs_coords': data_batch['lig_coords'][i],
                'ligs_coords_pred': data_sample['ligs_coords_pred'],
                'rec2_coords': rec2_coords,
                'rec2_coords_pred': rec2_coords_pred,
                'p2_rmsd_pred': data_sample['p2_rmsd_pred'],
            }
            self.results.append(result)
    
    def calculate(
        self,
        **kwargs
    ) -> dict:
        metrics = super().calculate(**kwargs)
        metrics['rec2_mean_rmsd'] = RMSD()(ligs_coords=kwargs['rec2_coords'], ligs_coords_pred=kwargs['rec2_coords_pred'])
        metrics['rec2_mean_mse'] = MSE()(kwargs['rec2_coords'], kwargs['rec2_coords_pred'])
        p2_rmsd = RMSD()(kwargs['rec2_coords'], kwargs['rec2_coords_pred'], reduction='none')
        metrics['p2_rmsd_mae'] = sum([MAE(i, j) for i, j in zip(p2_rmsd, kwargs['p2_rmsd_pred'])]) / len(p2_rmsd)
        return metrics
    