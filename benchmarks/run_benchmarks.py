"""
Benchmark Runner
================
Runs the de-obfuscator benchmark suite across 3 obfuscation levels.
Generates evaluation tables and exports results.
"""

import ast
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchmarks.obfuscate import Obfuscator
from benchmarks.metrics import compute_metrics
from src.parser import parse_source
from src.features import GraphFeatureExtractor
from src.reconstructor import PrettyPrinter


console = Console()
data_dir = Path(__file__).parent / "data"
clean_dir = data_dir / "clean"
obfuscated_dir = data_dir / "obfuscated"


def ensure_dirs():
    """Ensure data directories exist."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    obfuscated_dir.mkdir(parents=True, exist_ok=True)

    # If clean dir is empty, create a simple example
    if not list(clean_dir.glob("*.py")):
        sample = """
def calculate_factorial(number):
    if number == 0:
        return 1
    result = 1
    for index in range(1, number + 1):
        result = result * index
    return result
"""
        (clean_dir / "001_factorial.py").write_text(sample.strip())


def run_benchmark(level: int):
    """Run benchmark for a specific obfuscation level."""
    console.print(f"\n[bold green]Running Benchmark - Level {level}[/]")
    
    obfuscator = Obfuscator(level=level, seed=42)
    extractor = GraphFeatureExtractor()
    printer = PrettyPrinter()
    
    table = Table(title=f"Level {level} Results")
    table.add_column("File", style="cyan")
    table.add_column("Orig Score", justify="right")
    table.add_column("Obf Score", justify="right", style="red")
    table.add_column("Recovered Score", justify="right", style="green")
    table.add_column("Time (ms)", justify="right")
    
    total_orig = 0
    total_obf = 0
    total_rec = 0
    count = 0
    
    for file_path in sorted(clean_dir.glob("*.py")):
        source = file_path.read_text()
        
        # Original metrics
        orig_metrics = compute_metrics(source)
        
        # Obfuscate
        obfuscated = obfuscator.obfuscate(source)
        obf_file = obfuscated_dir / f"{file_path.stem}_L{level}.py"
        obf_file.write_text(obfuscated)
        obf_metrics = compute_metrics(obfuscated)
        
        # Pipeline: Parse -> Extract -> Reconstruct
        t0 = time.time()
        
        # Note: In a real run we'd pass obfuscated through the neural model.
        # For now, we simulate the pipeline by formatting the text and running the heuristics.
        graph = parse_source(obfuscated)
        features = extractor.extract(graph)
        recovered = printer.format_source(obfuscated)
        
        elapsed = (time.time() - t0) * 1000
        
        rec_metrics = compute_metrics(recovered)
        
        # Update aggregates
        total_orig += orig_metrics.overall_readability_score()
        total_obf += obf_metrics.overall_readability_score()
        total_rec += rec_metrics.overall_readability_score()
        count += 1
        
        table.add_row(
            file_path.name,
            f"{orig_metrics.overall_readability_score():.1f}",
            f"{obf_metrics.overall_readability_score():.1f}",
            f"{rec_metrics.overall_readability_score():.1f}",
            f"{elapsed:.1f}",
        )
        
    console.print(table)
    
    if count > 0:
        console.print(f"[bold]Average Original Readability:[/]  {total_orig/count:.1f}")
        console.print(f"[bold red]Average Obfuscated Readability:[/] {total_obf/count:.1f}")
        console.print(f"[bold green]Average Recovered Readability:[/]  {total_rec/count:.1f}")


if __name__ == "__main__":
    ensure_dirs()
    
    console.print("[bold]Neural De-obfuscator Benchmark Suite[/]")
    console.print("Testing pipeline on clean data corpus...\n")
    
    for lvl in [1, 2, 3]:
        run_benchmark(lvl)

