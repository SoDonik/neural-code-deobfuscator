"""
AST Parser Module
=================
Parses Python source code into a structured graph representation
using Python's built-in `ast` module. No `exec()` or `eval()` is used.

The graph representation captures:
  - Node types (FunctionDef, If, BinOp, etc.)
  - Parent-child edges (structural relationships)
  - Node depth and positional metadata
  - Variable name tokens for de-obfuscation training
"""

import ast
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────
# AST Node Wrapper
# ──────────────────────────────────────────────────────────────

# Canonical ordering of AST node types → integer IDs
NODE_TYPE_MAP = {
    "Module": 0, "FunctionDef": 1, "AsyncFunctionDef": 2, "ClassDef": 3,
    "Return": 4, "Delete": 5, "Assign": 6, "AugAssign": 7, "AnnAssign": 8,
    "For": 9, "AsyncFor": 10, "While": 11, "If": 12, "With": 13,
    "AsyncWith": 14, "Raise": 15, "Try": 16, "Assert": 17, "Import": 18,
    "ImportFrom": 19, "Global": 20, "Nonlocal": 21, "Expr": 22, "Pass": 23,
    "Break": 24, "Continue": 25,
    # Expressions
    "BoolOp": 26, "NamedExpr": 27, "BinOp": 28, "UnaryOp": 29,
    "Lambda": 30, "IfExp": 31, "Dict": 32, "Set": 33, "ListComp": 34,
    "SetComp": 35, "DictComp": 36, "GeneratorExp": 37, "Await": 38,
    "Yield": 39, "YieldFrom": 40, "Compare": 41, "Call": 42,
    "FormattedValue": 43, "JoinedStr": 44, "Constant": 45, "Attribute": 46,
    "Subscript": 47, "Starred": 48, "Name": 49, "List": 50, "Tuple": 51,
    "Slice": 52,
    # Misc
    "arg": 53, "arguments": 54, "keyword": 55, "alias": 56,
    "comprehension": 57, "ExceptHandler": 58,
    # Operators (encoded as pseudo-nodes)
    "Add": 59, "Sub": 60, "Mult": 61, "Div": 62, "Mod": 63,
    "Pow": 64, "FloorDiv": 65, "BitOr": 66, "BitXor": 67, "BitAnd": 68,
    "LShift": 69, "RShift": 70, "And": 71, "Or": 72, "Not": 73,
    "UAdd": 74, "USub": 75, "Invert": 76,
    "Eq": 77, "NotEq": 78, "Lt": 79, "LtE": 80, "Gt": 81, "GtE": 82,
    "Is": 83, "IsNot": 84, "In": 85, "NotIn": 86,
}

NUM_NODE_TYPES = len(NODE_TYPE_MAP)


@dataclass
class ASTNode:
    """Lightweight wrapper around a single AST node."""
    node_id: int                          # Unique ID within the graph
    node_type: str                        # e.g. "FunctionDef", "Name"
    node_type_id: int                     # Integer encoding of node_type
    depth: int                            # Depth in the tree (root = 0)
    name: Optional[str] = None            # Variable/function name if applicable
    value: Optional[str] = None           # Constant value if applicable
    lineno: int = 0                       # Source line number
    col_offset: int = 0                   # Source column offset
    children: list = field(default_factory=list)  # Child node IDs


# ──────────────────────────────────────────────────────────────
# AST Graph
# ──────────────────────────────────────────────────────────────

class ASTGraph:
    """
    A graph representation of a Python AST.

    Nodes are AST elements (statements, expressions, operators).
    Edges represent parent → child structural relationships.

    Attributes
    ----------
    nodes : list[ASTNode]
        All nodes in the graph.
    edges : list[tuple[int, int]]
        Directed edges (parent_id, child_id).
    name_tokens : list[str]
        All variable/function/class names found in the source.
    """

    def __init__(self):
        self.nodes: list[ASTNode] = []
        self.edges: list[tuple[int, int]] = []
        self.name_tokens: list[str] = []
        self._id_counter = 0

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def add_node(
        self,
        node_type: str,
        depth: int,
        name: Optional[str] = None,
        value: Optional[str] = None,
        lineno: int = 0,
        col_offset: int = 0,
    ) -> int:
        """Add a node to the graph and return its ID."""
        node_id = self._id_counter
        self._id_counter += 1

        type_id = NODE_TYPE_MAP.get(node_type, NUM_NODE_TYPES)

        node = ASTNode(
            node_id=node_id,
            node_type=node_type,
            node_type_id=type_id,
            depth=depth,
            name=name,
            value=value,
            lineno=lineno,
            col_offset=col_offset,
        )
        self.nodes.append(node)

        if name and node_type in ("Name", "FunctionDef", "ClassDef", "arg"):
            self.name_tokens.append(name)

        return node_id

    def add_edge(self, parent_id: int, child_id: int):
        """Add a directed edge from parent to child."""
        self.edges.append((parent_id, child_id))
        self.nodes[parent_id].children.append(child_id)

    def summary(self) -> dict:
        """Return a summary of the graph structure."""
        type_counts = {}
        for node in self.nodes:
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "max_depth": max(n.depth for n in self.nodes) if self.nodes else 0,
            "unique_names": len(set(self.name_tokens)),
            "type_distribution": dict(sorted(
                type_counts.items(), key=lambda x: -x[1]
            )[:10]),
        }


# ──────────────────────────────────────────────────────────────
# AST Visitor → Graph Builder
# ──────────────────────────────────────────────────────────────

class _GraphBuilder(ast.NodeVisitor):
    """Walks a Python AST and builds an ASTGraph."""

    def __init__(self):
        self.graph = ASTGraph()
        self._depth = 0
        self._parent_stack: list[int] = []

    def _get_node_name(self, node: ast.AST) -> Optional[str]:
        """Extract the identifier name from an AST node, if any."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.arg):
            return node.arg
        if isinstance(node, ast.alias):
            return node.asname or node.name
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _get_node_value(self, node: ast.AST) -> Optional[str]:
        """Extract a constant value from an AST node, if any."""
        if isinstance(node, ast.Constant):
            val = repr(node.value)
            return val[:64]  # Truncate to prevent huge strings
        return None

    def generic_visit(self, node: ast.AST):
        """Visit every node in the AST and add it to the graph."""
        node_type = type(node).__name__
        name = self._get_node_name(node)
        value = self._get_node_value(node)

        lineno = getattr(node, "lineno", 0)
        col_offset = getattr(node, "col_offset", 0)

        node_id = self.graph.add_node(
            node_type=node_type,
            depth=self._depth,
            name=name,
            value=value,
            lineno=lineno,
            col_offset=col_offset,
        )

        # Connect to parent
        if self._parent_stack:
            self.graph.add_edge(self._parent_stack[-1], node_id)

        # Recurse into children
        self._parent_stack.append(node_id)
        self._depth += 1

        for child in ast.iter_child_nodes(node):
            self.generic_visit(child)

        self._depth -= 1
        self._parent_stack.pop()


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def parse_source(source_code: str) -> ASTGraph:
    """
    Parse Python source code into an ASTGraph.

    Parameters
    ----------
    source_code : str
        Valid (or minified) Python source code.

    Returns
    -------
    ASTGraph
        A graph representation of the code's AST.

    Raises
    ------
    SyntaxError
        If the source code cannot be parsed.

    Examples
    --------
    >>> graph = parse_source("def f(x): return x * 2")
    >>> graph.num_nodes
    11
    >>> graph.name_tokens
    ['f', 'x', 'x']
    """
    tree = ast.parse(source_code)
    builder = _GraphBuilder()
    builder.generic_visit(tree)
    return builder.graph
