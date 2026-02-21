"""
Obfuscation Engine
==================
Generates obfuscated variants of clean Python code at three
difficulty levels for benchmarking the de-obfuscator.

Levels:
  1 (Light)  — Rename variables to single letters, strip comments
  2 (Medium) — + Inline functions, compress whitespace, merge with ;
  3 (Heavy)  — + Dead code injection, string encoding, control flow mangling
"""

import ast
import random
import re
import string
from typing import Optional


class Obfuscator:
    """
    Multi-level Python code obfuscator for benchmark generation.

    Parameters
    ----------
    level : int
        Obfuscation difficulty (1, 2, or 3).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, level: int = 1, seed: Optional[int] = None):
        assert level in (1, 2, 3), f"Level must be 1, 2, or 3, got {level}"
        self.level = level
        self.rng = random.Random(seed)

    def obfuscate(self, source: str) -> str:
        """
        Obfuscate Python source code.

        Parameters
        ----------
        source : str
            Clean Python source code.

        Returns
        -------
        str
            Obfuscated Python source code.
        """
        result = source

        # Level 1: Variable renaming + comment stripping
        result = self._strip_comments_and_docstrings(result)
        result = self._rename_variables(result)

        if self.level >= 2:
            # Level 2: Whitespace compression + statement merging
            result = self._compress_whitespace(result)
            result = self._merge_statements(result)

        if self.level >= 3:
            # Level 3: Dead code + string encoding
            result = self._inject_dead_code(result)
            result = self._encode_strings(result)

        return result

    # ──────────────────────────────────────────────────────────
    # Level 1: Renaming + Comment Stripping
    # ──────────────────────────────────────────────────────────

    def _strip_comments_and_docstrings(self, source: str) -> str:
        """Remove all comments and docstrings."""
        # Remove single-line comments
        source = re.sub(r"#[^\n]*", "", source)

        # Remove docstrings (triple-quoted strings at statement level)
        source = re.sub(r'"""[\s\S]*?"""', '""', source)
        source = re.sub(r"'''[\s\S]*?'''", "''", source)

        return source

    def _rename_variables(self, source: str) -> str:
        """Rename all user-defined variables to single letters."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        # Collect all user-defined names
        names = set()
        builtins_set = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        reserved = builtins_set | {"self", "cls", "__init__", "__main__", "__name__"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in reserved:
                names.add(node.id)
            elif isinstance(node, ast.FunctionDef) and node.name not in reserved:
                names.add(node.name)
            elif isinstance(node, ast.arg) and node.arg not in reserved:
                names.add(node.arg)

        # Generate single-letter replacements
        letters = list(string.ascii_lowercase)
        self.rng.shuffle(letters)

        rename_map = {}
        letter_idx = 0
        for name in sorted(names):
            if len(name) > 1:  # Only rename multi-letter names
                if letter_idx < len(letters):
                    rename_map[name] = letters[letter_idx]
                    letter_idx += 1
                else:
                    # Use two-letter combos when we run out
                    a = letters[letter_idx % 26]
                    b = letters[(letter_idx // 26) % 26]
                    rename_map[name] = a + b
                    letter_idx += 1

        # Apply renames
        result = source
        for old, new in sorted(rename_map.items(), key=lambda x: -len(x[0])):
            result = re.sub(rf"\b{re.escape(old)}\b", new, result)

        return result

    # ──────────────────────────────────────────────────────────
    # Level 2: Compression + Merging
    # ──────────────────────────────────────────────────────────

    def _compress_whitespace(self, source: str) -> str:
        """Remove unnecessary whitespace while preserving indentation."""
        lines = source.split("\n")
        compressed = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Preserve indentation but minimize internal spaces
            indent = len(line) - len(line.lstrip())
            # Reduce multiple spaces to single (outside strings)
            stripped = re.sub(r"  +", " ", stripped)
            compressed.append(" " * indent + stripped)
        return "\n".join(compressed)

    def _merge_statements(self, source: str) -> str:
        """Merge simple consecutive statements with semicolons."""
        lines = source.split("\n")
        merged = []
        i = 0

        while i < len(lines):
            line = lines[i]
            indent = len(line) - len(line.lstrip())

            # Try to merge with next line if both are simple statements
            if (
                i + 1 < len(lines)
                and self._is_simple_statement(line.strip())
                and self._is_simple_statement(lines[i + 1].strip())
                and indent == len(lines[i + 1]) - len(lines[i + 1].lstrip())
                and self.rng.random() < 0.5  # Only merge some of the time
            ):
                # Ensure no double semicolons
                first_part = line.rstrip(" ;")
                second_part = lines[i + 1].strip(" ;")
                merged.append(first_part + "; " + second_part)
                i += 2
            else:
                merged.append(line)
                i += 1

        return "\n".join(merged)

    @staticmethod
    def _is_simple_statement(line: str) -> bool:
        """Check if a line is a simple (non-compound) statement."""
        compound_keywords = {
            "if", "elif", "else", "for", "while", "def",
            "class", "try", "except", "finally", "with",
        }
        first_word = line.split()[0] if line.split() else ""
        return first_word not in compound_keywords and ":" not in line

    # ──────────────────────────────────────────────────────────
    # Level 3: Dead Code + String Encoding
    # ──────────────────────────────────────────────────────────

    def _inject_dead_code(self, source: str) -> str:
        """Insert unreachable dead code blocks using AST boundaries."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        # Find safe lines to inject after
        safe_lines = set()  # set of end_lineno
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                if hasattr(node, "end_lineno") and node.end_lineno is not None:
                    # Don't inject after break/continue/return, as following code is dead anyway
                    if isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
                        continue
                        
                    # Only inject after simple statements
                    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Pass, ast.AugAssign)):
                        safe_lines.add(node.end_lineno)

        dead_snippets = [
            ["if False:", "    _ = 0"],
            ["if 0:", "    pass"],
            ["_ = [i for i in range(0)]"],
            ["try:", "    _ = 1 / 1", "except:", "    pass"],
        ]

        lines = source.split("\n")
        result = []
        
        for i, line in enumerate(lines, 1):
            result.append(line)
            
            if i in safe_lines and self.rng.random() < 0.15:
                # Extract exact leading whitespace characters for the line itself
                indent = ""
                for char in line:
                    if char in " \t":
                        indent += char
                    else:
                        break
                        
                snippet_lines = self.rng.choice(dead_snippets)
                
                for sl in snippet_lines:
                    if sl.startswith("    "):
                        result.append(indent + sl)
                    else:
                        result.append(indent + sl)

        return "\n".join(result)

    def _encode_strings(self, source: str) -> str:
        """Replace string literals with encoded equivalents."""
        def encode_match(match):
            s = match.group(1)
            if len(s) < 2 or len(s) > 50:
                return match.group(0)
            # Replace with chr() concatenation
            encoded = "+".join(f"chr({ord(c)})" for c in s)
            return f"({encoded})"

        # Only encode double-quoted strings
        return re.sub(r'"([^"]*)"', encode_match, source)
