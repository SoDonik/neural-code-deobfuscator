"""Tests for the graph feature extractor."""

import pytest
import torch
from src.parser import parse_source
from src.features import GraphFeatureExtractor


class TestGraphFeatureExtractor:
    """Tests for GraphFeatureExtractor."""

    def setup_method(self):
        self.extractor = GraphFeatureExtractor(num_eigenvalues=16)

    def test_basic_extraction(self):
        graph = parse_source("def f(x): return x * 2")
        features = self.extractor.extract(graph)

        assert features.num_nodes > 0
        assert features.num_edges > 0
        assert features.eigenvalues.shape == (16,)
        assert features.feature_vector.ndim == 1

    def test_feature_vector_size(self):
        graph = parse_source("x = 1 + 2")
        features = self.extractor.extract(graph)
        assert len(features.feature_vector) == self.extractor.feature_dim

    def test_empty_graph(self):
        graph = parse_source("")
        features = self.extractor.extract(graph)
        assert features.num_nodes <= 1

    def test_betti_numbers(self):
        graph = parse_source("def f(x): return x")
        features = self.extractor.extract(graph)
        assert features.betti_0 >= 1.0  # At least 1 connected component

    def test_eigenvalues_normalized(self):
        graph = parse_source("""
def foo(x, y):
    if x > y:
        return x - y
    else:
        return y - x
""")
        features = self.extractor.extract(graph)
        # Normalized eigenvalues should be in [-1, 1]
        assert features.eigenvalues.abs().max() <= 1.0 + 1e-5

    def test_type_distribution_sums_to_one(self):
        graph = parse_source("for i in range(10): print(i)")
        features = self.extractor.extract(graph)
        total = features.type_distribution.sum().item()
        assert abs(total - 1.0) < 1e-5

    def test_complex_code_features(self):
        source = """
class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        result = 0
        for i in range(b):
            result = self.add(result, a)
        return result
"""
        graph = parse_source(source)
        features = self.extractor.extract(graph)
        assert features.max_depth > 3
        assert features.avg_branching_factor > 1.0
        assert features.num_unique_names > 3
