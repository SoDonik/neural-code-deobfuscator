"""
CLI Entry Point
===============
Command-line interface for the Neural Code De-obfuscator.

Usage:
    python -m src.cli deobfuscate input.py -o output.py
    python -m src.cli analyze input.py
    python -m src.cli benchmark --levels 1,2,3
"""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich import print as rprint

from src.parser import parse_source
from src.features import GraphFeatureExtractor
from src.reconstructor import PrettyPrinter


console = Console()


def cmd_analyze(args):
    """Analyze a Python file and display its AST graph features."""
    source = Path(args.input).read_text()

    console.print(Panel(
        f"[bold cyan]Analyzing:[/] {args.input}",
        title="Neural De-obfuscator",
        border_style="bright_cyan",
    ))

    # Parse AST
    t0 = time.time()
    graph = parse_source(source)
    parse_time = time.time() - t0

    # Extract features
    extractor = GraphFeatureExtractor()
    t1 = time.time()
    features = extractor.extract(graph)
    feat_time = time.time() - t1

    # Display results
    table = Table(title="Graph Analysis", border_style="bright_cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Nodes", str(features.num_nodes))
    table.add_row("Edges", str(features.num_edges))
    table.add_row("Max Depth", str(features.max_depth))
    table.add_row("Avg Branching Factor", f"{features.avg_branching_factor:.2f}")
    table.add_row("Unique Names", str(features.num_unique_names))
    table.add_row("Betti-0 (Components)", f"{features.betti_0:.0f}")
    table.add_row("Betti-1 (Cycles)", f"{features.betti_1:.0f}")
    table.add_row("Feature Vector Dim", str(len(features.feature_vector)))
    table.add_row("Parse Time", f"{parse_time*1000:.1f}ms")
    table.add_row("Feature Time", f"{feat_time*1000:.1f}ms")

    console.print(table)

    # Top eigenvalues
    top_k = min(8, len(features.eigenvalues))
    eig_str = ", ".join(f"{v:.4f}" for v in features.eigenvalues[:top_k])
    console.print(f"\n[bold]Top {top_k} Eigenvalues:[/] [{eig_str}]")

    # Name tokens
    names = graph.name_tokens[:20]
    console.print(f"[bold]Name Tokens:[/] {names}")

    # Summary
    summary = graph.summary()
    console.print(f"\n[bold]Type Distribution (top 10):[/]")
    for node_type, count in summary["type_distribution"].items():
        bar = "█" * min(count, 40)
        console.print(f"  {node_type:20s} {count:4d} {bar}")


def cmd_deobfuscate(args):
    """De-obfuscate a Python file using the pretty printer."""
    source = Path(args.input).read_text()

    console.print(Panel(
        f"[bold green]De-obfuscating:[/] {args.input}",
        title="Neural De-obfuscator",
        border_style="bright_green",
    ))

    # For now: use the heuristic pretty printer
    # (full model inference will be added after training)
    printer = PrettyPrinter(
        infer_names=not args.no_rename,
        max_line_length=args.line_length,
    )

    result = printer.format_source(source)

    if args.output:
        Path(args.output).write_text(result)
        console.print(f"[green]✓[/] Written to {args.output}")
    else:
        console.print("\n[bold]── Output ──[/]\n")
        console.print(Syntax(result, "python", theme="monokai", line_numbers=True))

    # Show before/after stats
    graph_before = parse_source(source)
    graph_after = parse_source(result)

    table = Table(title="Before / After", border_style="bright_green")
    table.add_column("Metric", style="bold")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")

    table.add_row("Lines", str(source.count("\n")), str(result.count("\n")))
    table.add_row("Characters", str(len(source)), str(len(result)))
    table.add_row("AST Nodes", str(graph_before.num_nodes), str(graph_after.num_nodes))
    table.add_row(
        "Unique Names",
        str(len(set(graph_before.name_tokens))),
        str(len(set(graph_after.name_tokens))),
    )
    console.print(table)


def cmd_benchmark(args):
    """Run the benchmark suite."""
    console.print(Panel(
        "[bold yellow]Running Benchmarks[/]",
        title="Neural De-obfuscator",
        border_style="bright_yellow",
    ))
    console.print("[yellow]Benchmark suite coming soon — run benchmarks/run_benchmarks.py[/]")


def main():
    parser = argparse.ArgumentParser(
        prog="neural-deobfuscator",
        description="Structure-Aware Neural Code De-obfuscator",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze AST structure")
    p_analyze.add_argument("input", help="Input Python file")
    p_analyze.set_defaults(func=cmd_analyze)

    # deobfuscate
    p_deobf = subparsers.add_parser("deobfuscate", help="De-obfuscate Python code")
    p_deobf.add_argument("input", help="Input Python file")
    p_deobf.add_argument("-o", "--output", help="Output file path")
    p_deobf.add_argument("--no-rename", action="store_true", help="Disable name inference")
    p_deobf.add_argument("--line-length", type=int, default=88, help="Max line length")
    p_deobf.set_defaults(func=cmd_deobfuscate)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmarks")
    p_bench.add_argument("--levels", default="1,2,3", help="Obfuscation levels")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
