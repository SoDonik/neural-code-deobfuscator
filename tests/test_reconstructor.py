"""Tests for the pretty printer / code reconstructor."""

import pytest
from src.reconstructor import PrettyPrinter


class TestPrettyPrinter:
    """Tests for PrettyPrinter."""

    def setup_method(self):
        self.printer = PrettyPrinter(infer_names=True)

    def test_basic_formatting(self):
        ugly = "x=1;y=2"
        result = self.printer.format_source(ugly)
        assert "x" in result or "value" in result

    def test_name_inference(self):
        source = "def f(x):\n    return x * 2\n"
        result = self.printer.format_source(source)
        # Should attempt to rename single-letter variables
        assert len(result) > 0

    def test_preserves_valid_syntax(self):
        source = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
        result = self.printer.format_source(source)
        # Result should still be valid Python
        compile(result, "<test>", "exec")

    def test_handles_empty_input(self):
        result = self.printer.format_source("")
        assert isinstance(result, str)

    def test_removes_trailing_whitespace(self):
        source = "x = 1   \ny = 2   \n"
        result = self.printer.format_source(source)
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_no_rename_option(self):
        printer = PrettyPrinter(infer_names=False)
        source = "x = 1\n"
        result = printer.format_source(source)
        assert "x" in result

    def test_token_reconstruction(self):
        # Simulate model output: byte tokens for "def f(x): return x"
        tokens = list(b"def f(x): return x\n")
        result = self.printer.reconstruct(tokens)
        assert "def" in result
        assert "return" in result

    def test_handles_malformed_code(self):
        # Should not crash on broken code
        source = "def broken(\n    x = {\n"
        result = self.printer.format_source(source)
        assert isinstance(result, str)
