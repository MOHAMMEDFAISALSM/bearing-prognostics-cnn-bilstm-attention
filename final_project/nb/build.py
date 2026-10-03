import json, sys, importlib
sys.path.insert(0, "nb")
import nbdefs
for part in ("part1", "part2", "part3"):
    importlib.import_module(part)
nb = {"cells": nbdefs.CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = sys.argv[1]
json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print(len(nbdefs.CELLS), "cells ->", out)
