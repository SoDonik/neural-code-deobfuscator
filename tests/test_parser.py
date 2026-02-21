"""Tests for the AST parser module."""

import pytest
from src.parser import parse_source, ASTGraph


class TestParseSource:
    """Tests for parse_source()."""

    def test_simple_function(self):
        source = "def f(x): return x * 2"
        graph = parse_source(source)
        assert graph.num_nodes > 0
        assert graph.num_edges > 0
        assert "f" in graph.name_tokens
        assert "x" in graph.name_tokens

    def test_class_definition(self):
        source = """
class Foo:
    def __init__(self, x):
        self.x = x

    def bar(self):
        return self.x + 1
"""
        graph = parse_source(source)
        assert graph.num_nodes > 10
        assert "Foo" in graph.name_tokens
        assert "bar" in graph.name_tokens

    def test_empty_source(self):
        graph = parse_source("")
        assert graph.num_nodes == 1  # Module node
        assert graph.num_edges == 0

    def test_complex_code(self):
        source = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n + 1):
        a, b = b, a + b
    return b
"""
        graph = parse_source(source)
        assert graph.num_nodes > 20
        summary = graph.summary()
        assert summary["num_nodes"] > 20
        assert summary["max_depth"] > 3

    def test_minified_code(self):
        source = "x=1;y=2;z=x+y;print(z)"
        graph = parse_source(source)
        assert graph.num_nodes > 0

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            parse_source("def (broken syntax")

    def test_node_types_are_encoded(self):
        source = "x = 42"
        graph = parse_source(source)
        # All nodes should have valid type IDs
        for node in graph.nodes:
            assert node.node_type_id >= 0

    def test_edges_are_valid(self):
        source = "def f(x): return x"
        graph = parse_source(source)
        for parent_id, child_id in graph.edges:
            assert 0 <= parent_id < graph.num_nodes
            assert 0 <= child_id < graph.num_nodes
