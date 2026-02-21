import ast
from benchmarks.obfuscate import Obfuscator
from pathlib import Path

obf = Obfuscator(level=3, seed=42)
for f in Path("benchmarks/data/clean").glob("*.py"):
    source = f.read_text()
    try:
        obfuscated = obf.obfuscate(source)
        ast.parse(obfuscated)
    except Exception as e:
        print(f"FAILED on {f.name}: {e}")
        # print the lines around the error
        lines = obfuscated.split('\n')
        for i, l in enumerate(lines):
            print(f"{i+1:3d} | {l}")
        break
