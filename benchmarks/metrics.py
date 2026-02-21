"""
Code Readability Metrics
========================
Implements standard software complexity metrics for measuring
how readable/obfuscated a piece of code is.

Metrics:
  - Halstead Complexity (vocabulary, difficulty, effort)
  - Cyclomatic Complexity
  - Lines of Code statistics
  - Name Descriptiveness Score
"""

import ast
import math
import re
import keyword
from dataclasses import dataclass


@dataclass
class ReadabilityMetrics:
    """Container for all computed readability metrics."""

    # Halstead metrics
    halstead_vocabulary: int       # Number of distinct operators + operands
    halstead_length: int           # Total operators + operands
    halstead_difficulty: float     # How hard the code is to understand
    halstead_effort: float         # Mental effort to comprehend
    halstead_volume: float         # Information content

    # Cyclomatic complexity
    cyclomatic_complexity: int     # Number of independent paths

    # Line statistics
    total_lines: int
    code_lines: int                # Non-blank, non-comment lines
    avg_line_length: float
    max_line_length: int

    # Name quality
    name_descriptiveness: float    # 0.0 (all single-letter) to 1.0 (all descriptive)
    avg_name_length: float
    single_letter_ratio: float     # Fraction of names that are single letters

    def overall_readability_score(self) -> float:
        """
        Compute a composite readability score from 0 (unreadable) to 100 (perfect).

        Higher = more readable.
        """
        # Normalize components to [0, 1] where 1 = good
        halstead_score = max(0, 1.0 - self.halstead_difficulty / 100.0)
        cyclomatic_score = max(0, 1.0 - self.cyclomatic_complexity / 50.0)
        name_score = self.name_descriptiveness
        line_score = max(0, 1.0 - self.avg_line_length / 120.0)

        # Weighted combination
        score = (
            0.30 * halstead_score +
            0.20 * cyclomatic_score +
            0.35 * name_score +
            0.15 * line_score
        )

        return round(score * 100, 1)


def compute_metrics(source: str) -> ReadabilityMetrics:
    """
    Compute all readability metrics for a piece of Python source code.

    Parameters
    ----------
    source : str
        Python source code to analyze.

    Returns
    -------
    ReadabilityMetrics
        All computed metrics.
    """
    lines = source.split("\n")

    # ── Line statistics ──
    total_lines = len(lines)
    code_lines = sum(
        1 for line in lines
        if line.strip() and not line.strip().startswith("#")
    )
    line_lengths = [len(line) for line in lines if line.strip()]
    avg_line_length = sum(line_lengths) / max(1, len(line_lengths))
    max_line_length = max(line_lengths) if line_lengths else 0

    # ── Halstead metrics ──
    operators, operands = _extract_halstead_tokens(source)

    n1 = len(set(operators))  # Distinct operators
    n2 = len(set(operands))   # Distinct operands
    N1 = len(operators)       # Total operators
    N2 = len(operands)        # Total operands

    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * math.log2(max(2, vocabulary))
    difficulty = (n1 / 2.0) * (N2 / max(1, n2))
    effort = difficulty * volume

    # ── Cyclomatic complexity ──
    cyclomatic = _compute_cyclomatic(source)

    # ── Name quality metrics ──
    names = _extract_names(source)
    name_lengths = [len(n) for n in names]
    avg_name_length = sum(name_lengths) / max(1, len(name_lengths))
    single_letter = sum(1 for n in names if len(n) == 1)
    single_letter_ratio = single_letter / max(1, len(names))
    name_descriptiveness = 1.0 - single_letter_ratio

    return ReadabilityMetrics(
        halstead_vocabulary=vocabulary,
        halstead_length=length,
        halstead_difficulty=round(difficulty, 2),
        halstead_effort=round(effort, 2),
        halstead_volume=round(volume, 2),
        cyclomatic_complexity=cyclomatic,
        total_lines=total_lines,
        code_lines=code_lines,
        avg_line_length=round(avg_line_length, 1),
        max_line_length=max_line_length,
        name_descriptiveness=round(name_descriptiveness, 3),
        avg_name_length=round(avg_name_length, 1),
        single_letter_ratio=round(single_letter_ratio, 3),
    )


def _extract_halstead_tokens(source: str):
    """Extract Halstead operators and operands from source code."""
    operators = []
    operands = []

    # Python operators
    op_pattern = re.compile(
        r"(\+\+|--|<<|>>|<=|>=|==|!=|&&|\|\||"
        r"[+\-*/%&|^~<>=!]|and|or|not|in|is)"
    )

    # Tokenize
    tokens = re.findall(r"[a-zA-Z_]\w*|[0-9]+\.?[0-9]*|[^\s\w]", source)

    py_keywords = set(keyword.kwlist)

    for token in tokens:
        if token in py_keywords or op_pattern.match(token):
            operators.append(token)
        elif re.match(r"[a-zA-Z_]\w*", token):
            operands.append(token)
        elif re.match(r"[0-9]", token):
            operands.append(token)
        elif token in "()[]{}:;,.":
            operators.append(token)

    return operators, operands


def _compute_cyclomatic(source: str) -> int:
    """Compute McCabe cyclomatic complexity."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 1

    complexity = 1  # Base complexity

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Each 'and'/'or' adds a branch
            complexity += len(node.values) - 1
        elif isinstance(node, ast.Assert):
            complexity += 1

    return complexity


def _extract_names(source: str) -> list[str]:
    """Extract all user-defined variable/function names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    names = []
    builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    reserved = builtins_set | set(keyword.kwlist) | {"self", "cls"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in reserved:
            names.append(node.id)
        elif isinstance(node, ast.FunctionDef) and node.name not in reserved:
            names.append(node.name)
        elif isinstance(node, ast.arg) and node.arg not in reserved:
            names.append(node.arg)

    return names
