"""
Graph Feature Extractor
=======================
Extracts numerical feature vectors from an ASTGraph for use
as input to the neural de-obfuscation model.

Features extracted:
  - Adjacency matrix (sparse COO/CSR via SciPy)
  - Graph Laplacian matrix (sparse)
  - Spectral features (eigenvalues of the Laplacian)
  - Betti numbers (connected components, cycles)
  - Node type distribution vector
  - Structural statistics (depth, branching factor, etc.)
"""

import math
from dataclasses import dataclass

import torch
import numpy as np
from scipy.sparse import coo_matrix, eye as speye
from scipy.sparse.csgraph import laplacian as sparse_laplacian
from scipy.sparse.linalg import eigsh

from src.parser.ast_parser import ASTGraph, NUM_NODE_TYPES


@dataclass
class GraphFeatures:
    """Container for all features extracted from an AST graph."""

    # Spectral features
    eigenvalues: torch.Tensor           # (k,) top-k Laplacian eigenvalues
    betti_0: float                      # Number of connected components
    betti_1: float                      # Approximate number of independent cycles

    # Structural features
    num_nodes: int
    num_edges: int
    max_depth: int
    avg_branching_factor: float
    num_unique_names: int

    # Node type distribution
    type_distribution: torch.Tensor     # (NUM_NODE_TYPES,) normalized histogram

    # Combined fixed-size vector for the model
    feature_vector: torch.Tensor        # (feature_dim,) concatenated features


class GraphFeatureExtractor:
    """
    Extracts a fixed-size feature vector from an ASTGraph.

    Parameters
    ----------
    num_eigenvalues : int
        Number of Laplacian eigenvalues to keep (default: 32).
    """

    def __init__(self, num_eigenvalues: int = 32):
        self.num_eigenvalues = num_eigenvalues

    def extract(self, graph: ASTGraph) -> GraphFeatures:
        """
        Extract all features from an AST graph.

        Parameters
        ----------
        graph : ASTGraph
            The AST graph to extract features from.

        Returns
        -------
        GraphFeatures
            All computed features, including a concatenated feature vector.
        """
        n = graph.num_nodes

        if n == 0:
            return self._empty_features()

        # ── Build sparse adjacency matrix (undirected) ──
        if graph.edges:
            rows, cols = zip(*graph.edges)
            # Make undirected: add both directions
            rows = list(rows) + list(cols)
            cols = list(cols) + list(rows[:len(graph.edges)])
            data = [1.0] * len(rows)
            adj_sparse = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        else:
            adj_sparse = coo_matrix((n, n)).tocsr()

        # ── Spectral decomposition ──
        # Use sparse eigsh for large graphs, dense fallback for small ones
        if n >= 50:
            # Sparse path: compute Laplacian and top-k eigenvalues
            lap = sparse_laplacian(adj_sparse, normed=False)
            # eigsh needs k < n; request smallest eigenvalues (shift-invert)
            k = min(self.num_eigenvalues, n - 1)
            try:
                eigs = eigsh(lap, k=k, which='SM', return_eigenvectors=False)
                eigenvalues_np = np.sort(eigs)
            except Exception:
                # Fallback to dense if sparse solver fails
                lap_dense = lap.toarray()
                eigenvalues_np = np.sort(np.linalg.eigvalsh(lap_dense))[:k]
        else:
            # Dense path for small graphs (sparse solvers less stable)
            adj_dense = adj_sparse.toarray()
            degree = np.diag(adj_dense.sum(axis=1))
            lap_dense = degree - adj_dense + np.eye(n) * 1e-6
            all_eigs = np.sort(np.linalg.eigvalsh(lap_dense))
            k = min(self.num_eigenvalues, n)
            eigenvalues_np = all_eigs[:k]

        top_eigenvalues = torch.from_numpy(eigenvalues_np.astype(np.float32))

        # Pad to fixed size if needed
        if len(top_eigenvalues) < self.num_eigenvalues:
            padding = torch.zeros(self.num_eigenvalues - len(top_eigenvalues))
            top_eigenvalues = torch.cat([top_eigenvalues, padding])

        # ── Betti numbers (topological features) ──
        # Betti-0: number of connected components
        #   = number of near-zero eigenvalues (computed BEFORE normalization)
        betti_0 = float((top_eigenvalues.abs() < 1e-4).sum())

        # Normalize eigenvalues
        max_eig = top_eigenvalues.abs().max()
        if max_eig > 0:
            top_eigenvalues = top_eigenvalues / max_eig

        # Betti-1: approximate number of independent cycles
        #   = edges - nodes + connected_components (Euler characteristic)
        betti_1 = max(0.0, graph.num_edges - graph.num_nodes + betti_0)

        # ── Structural statistics ──
        max_depth = max(node.depth for node in graph.nodes)
        children_counts = [len(node.children) for node in graph.nodes]
        avg_branching = (
            sum(children_counts) / max(1, len([c for c in children_counts if c > 0]))
        )
        num_unique_names = len(set(graph.name_tokens))

        # ── Node type distribution ──
        type_hist = torch.zeros(NUM_NODE_TYPES + 1)  # +1 for unknown types
        for node in graph.nodes:
            idx = min(node.node_type_id, NUM_NODE_TYPES)
            type_hist[idx] += 1.0

        # Normalize to probability distribution
        total = type_hist.sum()
        if total > 0:
            type_hist = type_hist / total

        # ── Concatenate into a single feature vector ──
        structural = torch.tensor([
            float(n),
            float(graph.num_edges),
            float(max_depth),
            avg_branching,
            float(num_unique_names),
            betti_0,
            betti_1,
            # Normalized versions
            math.log1p(n),
            math.log1p(graph.num_edges),
            max_depth / max(1.0, n),
        ])

        feature_vector = torch.cat([
            top_eigenvalues,        # (num_eigenvalues,)
            structural,             # (10,)
            type_hist,              # (NUM_NODE_TYPES + 1,)
        ])

        return GraphFeatures(
            eigenvalues=top_eigenvalues,
            betti_0=betti_0,
            betti_1=betti_1,
            num_nodes=n,
            num_edges=graph.num_edges,
            max_depth=max_depth,
            avg_branching_factor=avg_branching,
            num_unique_names=num_unique_names,
            type_distribution=type_hist,
            feature_vector=feature_vector,
        )

    def _empty_features(self) -> GraphFeatures:
        """Return zeroed features for an empty graph."""
        return GraphFeatures(
            eigenvalues=torch.zeros(self.num_eigenvalues),
            betti_0=0.0,
            betti_1=0.0,
            num_nodes=0,
            num_edges=0,
            max_depth=0,
            avg_branching_factor=0.0,
            num_unique_names=0,
            type_distribution=torch.zeros(NUM_NODE_TYPES + 1),
            feature_vector=torch.zeros(
                self.num_eigenvalues + 10 + NUM_NODE_TYPES + 1
            ),
        )

    @property
    def feature_dim(self) -> int:
        """Total dimensionality of the concatenated feature vector."""
        return self.num_eigenvalues + 10 + NUM_NODE_TYPES + 1
