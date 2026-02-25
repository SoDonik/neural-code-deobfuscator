import ast
from benchmarks.obfuscate import Obfuscator
from pathlib import Path

obf = Obfuscator(level=3, seed=42)
for file_path in sorted(Path("benchmarks/data/clean").glob("*.py")):
    source = file_path.read_text()
    obfuscated = obf.obfuscate(source)
    try:
        ast.parse(obfuscated)
    except Exception as e:
        print(f"\nFAILED on {file_path.name}: {e}")
        print("--- OBFUSCATED ---")
        for i, line in enumerate(obfuscated.split('\n')):
            print(f"{i+1:3d} | {line}")
        break
