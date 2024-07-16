"""visualization of t-SNE of protein feature"""
import os

import dgl
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from tqdm import tqdm
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
from matplotlib.colors import ListedColormap

from mmpretrain.apis.model import get_model
from mmpretrain.registry import DATASETS

seaborn.set(style='ticks')
custom_cmap = ListedColormap(['#3e55a5', '#ed2024', '#69bd45'])


def move_to_cuda(x):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    elif isinstance(x, dgl.DGLGraph):
        return x.to('cuda')
    elif isinstance(x, dict):
        return {k: move_to_cuda(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_cuda(v) for v in x]
    else:
        return x


def extract_features():
    save_file = 'output/tsne/encoder_features_PROTAC_all_merge.pth'
    if os.path.exists(save_file):
        return torch.load(save_file)

    config = 'output/Glue2_DecD4/glue.py'
    checkpoint = 'output/Glue2_DecD4/epoch_1000.pth'
    cfg = Config.fromfile(config)
    model: torch.nn.Module = get_model(
        config, pretrained=checkpoint, device='cuda')
    model.eval()

    # build dataloader
    dataloader_cfg = cfg.get('test_dataloader')
    dataset_cfg = dataloader_cfg['dataset']

    dataset_cfg.preprocess_sub_path = 'pdb2311_merge'
    # dataset_cfg.complex_names_path = 'data/PROTAC/complex.txt'
    dataset_cfg.complex_names_path = 'output/tsne/all.names'

    dataset = DATASETS.build(dataset_cfg)
    dataloader_cfg.dataset = dataset
    dataloader = Runner.build_dataloader(dataloader_cfg)

    # get encoder features
    lig_features = []
    p1_features = []
    p2_features = []
    for data in tqdm(dataloader):
        with torch.no_grad():
            # preprocess
            data = model.data_preprocessor(data)
            data = move_to_cuda(data)

            # extract encoder features
            lig_graph, p1_graph, p2_graph = model.forward_encoder(
                **data)  # Tuple(lig_graph, p1_graph, p2_graph)

            # post process
            lig_graph = dgl.unbatch(lig_graph.cpu())
            lig_feats = [g.ndata['feat'].mean(0).numpy() for g in lig_graph]
            p1_graph = dgl.unbatch(p1_graph.cpu())
            p1_feats = [g.ndata['feat'].mean(0).numpy() for g in p1_graph]
            p2_graph = dgl.unbatch(p2_graph.cpu())
            p2_feats = [g.ndata['feat'].mean(0).numpy() for g in p2_graph]

        # save batch features
        lig_features.extend(lig_feats)
        p1_features.extend(p1_feats)
        p2_features.extend(p2_feats)

    assert len(lig_features) == len(p1_features) == len(p2_features)
    # save features
    data = {
        'lig_features': lig_features,
        'p1_features': p1_features,
        'p2_features': p2_features
    }
    torch.save(data, save_file)
    return data


def main(learning_rate):
    results: dict = extract_features()
    labels = np.loadtxt('output/tsne/all.names', dtype=str, delimiter=',')
    labels = labels[:, 1].astype(int)
    classes = ['Training', 'PROTAC', 'MGD']

    # build t-SNE model
    tsne_model = TSNE(
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate=learning_rate,    # [10, 1000]
        n_iter=1000,
        n_iter_without_progress=300,
        init='random')

    # draw train first, protac last
    sort_idx = labels.argsort()
    labels = labels[sort_idx]

    p1_p2 = np.concatenate([results['p1_features'], results['p2_features']], axis=1)
    results = {
        'p1_p2': p1_p2,
    }

    # run and get results
    # logger.info('Running t-SNE.')
    for key, val in results.items():
        val = np.array(val)
        print('value shape:', val.shape)
        print('start tsne')
        result = tsne_model.fit_transform(val)
        res_min, res_max = result.min(0), result.max(0)
        res_norm = (result - res_min) / (res_max - res_min)
        np.savetxt(f'output/tsne/tsne_{key}_lr{learning_rate}.csv', res_norm, delimiter=',')
        print('start plot')
        # sort by labels
        res_norm = res_norm[sort_idx]

        _, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            res_norm[:, 0],
            res_norm[:, 1],
            alpha=1.0,
            s=15,
            c=labels,
            # cmap='tab10',
            cmap=custom_cmap
            )
        legend = ax.legend(scatter.legend_elements()[0], classes)
        ax.add_artist(legend)

        # scatter = seaborn.scatterplot(
        #     x=res_norm[:, 0],
        #     y=res_norm[:, 1],
        #     hue=labels,
        #     palette='tab10',
        #     ax=ax)
        # handles, _ = scatter.get_legend_handles_labels()
        # legend = plt.legend(handles, classes)

        plt.savefig(f'output/tsne/tsne_{key}_lr{learning_rate}.pdf')


if __name__ == '__main__':
    # for lr in (10, 50, 100, 200, 400, 600, 800, 1000):
    for lr in (50, ):
        main(lr)
