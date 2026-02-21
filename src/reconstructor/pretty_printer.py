"""
Pretty Printer / Code Reconstructor
====================================
Converts model output (byte-level token IDs) back into clean,
readable, PEP 8-compliant Python source code.

Pipeline:
  1. Decode byte tokens → raw string
  2. Parse with `ast` to validate syntax
  3. Apply formatting rules (indentation, spacing, line length)
  4. Attempt to infer meaningful variable names from context
"""

import ast
import keyword
import re
from typing import Optional


# ──────────────────────────────────────────────────────────────
# Name Inference Heuristics
# ──────────────────────────────────────────────────────────────

# Common single-letter → descriptive name mappings
_NAME_HEURISTICS = {
    # Loop variables
    "i": "index",
    "j": "inner_index",
    "k": "key",
    "n": "count",
    "m": "size",
    # Data
    "x": "value",
    "y": "result",
    "z": "output",
    "s": "text",
    "d": "data",
    "v": "val",
    "w": "weight",
    "a": "first",
    "b": "second",
    "c": "third",
    # Control
    "f": "func",
    "g": "gen",
    "p": "param",
    "q": "queue",
    "r": "res",
    "t": "temp",
    "e": "element",
    "l": "items",  # noqa: E741
}

# Context-aware name patterns
_CONTEXT_PATTERNS = {
    r"for\s+(\w)\s+in\s+range": {"i": "index", "j": "step", "k": "iter_var"},
    r"for\s+(\w)\s+in\s+\w+": {"x": "item", "v": "val", "e": "element"},
    r"(\w)\s*=\s*\[\]": {"a": "items", "r": "results", "l": "values"},
    r"(\w)\s*=\s*\{\}": {"d": "mapping", "r": "lookup", "c": "cache"},
    r"(\w)\s*=\s*0": {"c": "counter", "s": "total", "n": "count"},
    r"def\s+\w+\((\w)": {"s": "self", "x": "input_val", "n": "num"},
}


class PrettyPrinter:
    """
    Converts raw model output into clean Python code.

    Parameters
    ----------
    infer_names : bool
        Whether to attempt renaming single-letter variables
        to more descriptive names (default: True).
    max_line_length : int
        Target maximum line length for formatting (default: 88).
    indent_size : int
        Number of spaces per indentation level (default: 4).
    """

    def __init__(
        self,
        infer_names: bool = True,
        max_line_length: int = 88,
        indent_size: int = 4,
    ):
        self.infer_names = infer_names
        self.max_line_length = max_line_length
        self.indent_size = indent_size

    def reconstruct(
        self,
        tokens: list[int],
        original_source: Optional[str] = None,
    ) -> str:
        """
        Convert byte-level token IDs back to clean Python source.

        Parameters
        ----------
        tokens : list[int]
            Byte-level token IDs from the model output.
        original_source : str, optional
            The original obfuscated source (for context-aware name inference).

        Returns
        -------
        str
            Cleaned, formatted Python source code.
        """
        # Step 1: Decode bytes → string
        raw = bytes(t for t in tokens if 0 < t < 256).decode(
            "utf-8", errors="replace"
        )

        # Step 2: Basic cleanup
        raw = self._clean_raw_output(raw)

        # Step 3: Validate and fix syntax
        raw = self._fix_syntax(raw)

        # Step 4: Apply formatting
        raw = self._format_code(raw)

        # Step 5: Infer better variable names
        if self.infer_names:
            raw = self._infer_variable_names(raw, original_source)

        return raw

    def format_source(self, source: str) -> str:
        """
        Format existing Python source code (without model decoding).

        Parameters
        ----------
        source : str
            Python source code to format.

        Returns
        -------
        str
            Formatted Python source code.
        """
        source = self._clean_raw_output(source)
        source = self._format_code(source)
        if self.infer_names:
            source = self._infer_variable_names(source)
        return source

    # ──────────────────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────────────────

    def _clean_raw_output(self, text: str) -> str:
        """Remove common artifacts from model output."""
        # Strip null bytes and control characters
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]

        # Remove excessive blank lines (max 2 consecutive)
        cleaned = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    cleaned.append(line)
            else:
                blank_count = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip() + "\n"

    def _fix_syntax(self, source: str) -> str:
        """Attempt to fix common syntax issues in model output."""
        try:
            ast.parse(source)
            return source  # Already valid
        except SyntaxError:
            pass

        # Try common fixes
        fixes = [
            # Missing colons after control flow
            (r"((?:if|elif|else|for|while|def|class|try|except|finally|with)\s+[^\n:]+)\n",
             r"\1:\n"),
            # Unclosed parentheses at end
            (r"([^)]*\([^)]*$)", r"\1)"),
            # Unclosed brackets at end
            (r"([^\]]*\[[^\]]*$)", r"\1]"),
        ]

        fixed = source
        for pattern, replacement in fixes:
            try:
                fixed = re.sub(pattern, replacement, fixed)
                ast.parse(fixed)
                return fixed
            except (SyntaxError, re.error):
                fixed = source

        return source  # Return original if no fix works

    def _format_code(self, source: str) -> str:
        """Apply PEP 8-style formatting rules."""
        lines = source.split("\n")
        formatted = []

        for line in lines:
            # Normalize indentation to spaces
            stripped = line.lstrip()
            if not stripped:
                formatted.append("")
                continue

            # Count leading whitespace
            indent_chars = len(line) - len(stripped)
            # Convert tabs to spaces
            indent = line[:indent_chars].replace("\t", " " * self.indent_size)
            indent_level = len(indent)

            # Add spaces around operators
            stripped = re.sub(r"(\w)([=!<>]+)(\w)", r"\1 \2 \3", stripped)
            # But fix double-spacing
            stripped = re.sub(r"  +", " ", stripped)
            # Fix spacing after commas
            stripped = re.sub(r",(\S)", r", \1", stripped)

            formatted.append(" " * indent_level + stripped)

        return "\n".join(formatted)

    def _infer_variable_names(
        self,
        source: str,
        original: Optional[str] = None,
    ) -> str:
        """
        Attempt to rename single-letter variables to descriptive names.

        Uses heuristic rules and context patterns to infer what
        obfuscated variable names likely represent.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        # Collect all single-letter variable names
        single_letter_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and len(node.id) == 1:
                # Don't rename if it's a Python keyword or builtin
                if node.id not in keyword.kwlist:
                    single_letter_names.add(node.id)

        if not single_letter_names:
            return source

        # Build rename map using heuristics
        rename_map = {}
        used_names = set()

        for name in sorted(single_letter_names):
            # Try context-aware patterns first
            new_name = self._infer_from_context(name, source)

            # Fall back to generic heuristics
            if not new_name and name in _NAME_HEURISTICS:
                new_name = _NAME_HEURISTICS[name]

            if new_name and new_name not in used_names:
                rename_map[name] = new_name
                used_names.add(new_name)

        # Apply renames carefully (only whole-word matches)
        result = source
        for old_name, new_name in sorted(
            rename_map.items(), key=lambda x: -len(x[0])
        ):
            # Use word boundary regex to avoid partial replacements
            pattern = rf"\b{re.escape(old_name)}\b"
            result = re.sub(pattern, new_name, result)

        # Validate the renamed code still parses
        try:
            ast.parse(result)
            return result
        except SyntaxError:
            return source  # Revert if renaming broke something

    def _infer_from_context(self, name: str, source: str) -> Optional[str]:
        """Infer a variable name from its usage context."""
        for pattern, name_map in _CONTEXT_PATTERNS.items():
            if name in name_map and re.search(pattern, source):
                return name_map[name]
        return None
