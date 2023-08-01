# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import os.path as osp
import torch_geometric.transforms as T

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from utils.classic import Planetoid
from definitions import ROOT_DIR


class Actor(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/actor.py

    The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # Remove self-loops
            edge_index, _ = remove_self_loops(edge_index)
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/wikipedia_network.py

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
        x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split('\t')[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/webkb.py

    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           '1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


def get_fixed_splits(data, dataset_name, seed):
    if dataset_name not in ["roman-empire", "rmazon-ratings", "minesweeper", "tolokers", "questions"]:
        with np.load(f'splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']

        data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        if dataset_name in {'cora', 'citeseer', 'pubmed'}:
            data.train_mask[data.non_valid_samples] = False
            data.test_mask[data.non_valid_samples] = False
            data.val_mask[data.non_valid_samples] = False
            print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
            print("Nodes", data.x.size(0))
            print("Non valid", len(data.non_valid_samples))
        else:
            assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)
    else:
        data.train_mask = data.train_masks[:, seed]
        data.val_mask = data.val_masks[:, seed]
        data.test_mask = data.test_masks[:, seed]
    return data

class HeterophilousGraphDataset(InMemoryDataset):
    r"""The heterophilous graphs :obj:`"Roman-empire"`,
    :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"` and
    :obj:`"Questions"` from the `"A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?"
    <https://arxiv.org/abs/2302.11640>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"Roman-empire"`,
            :obj:`"Amazon-ratings"`, :obj:`"Minesweeper"`, :obj:`"Tolokers"`,
            :obj:`"Questions"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Roman-empire
          - 22,662
          - 32,927
          - 300
          - 18
        * - Amazon-ratings
          - 24,492
          - 93,050
          - 300
          - 5
        * - Minesweeper
          - 10,000
          - 39,402
          - 7
          - 2
        * - Tolokers
          - 11,758
          - 519,000
          - 10
          - 2
        * - Questions
          - 48,921
          - 153,540
          - 301
          - 2
    """
    url = ('https://github.com/yandex-research/heterophilous-graphs/raw/'
           'main/data')

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower().replace('-', '_')
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
        ]

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(f'{self.url}/{self.name}.npz', self.raw_dir)

    def process(self):
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name})'


def get_dataset(name):
    data_root = osp.join(ROOT_DIR, 'datasets')
    if name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == 'film':
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures())
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]:
        transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
        dataset = HeterophilousGraphDataset(root=data_root, name=name, transform=transform)
        dataset.data.train_masks = dataset.data.train_mask.clone()
        dataset.data.val_masks = dataset.data.val_mask.clone()
        dataset.data.test_masks = dataset.data.test_mask.clone()
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset