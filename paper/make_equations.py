"""Converts equations.tex.json (LaTeX) to native Word math (OMML) via pandoc -> equations.omml.json.
build_paper.js splices these into the manuscript, so every equation stays editable in Word."""
import json
import os
import re
import subprocess
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
eqs = json.load(open(os.path.join(HERE, "equations.tex.json"), encoding="utf-8"))
with tempfile.TemporaryDirectory() as tmp:
    md, out = os.path.join(tmp, "eqs.md"), os.path.join(tmp, "eqs.docx")
    open(md, "w", encoding="utf-8").write("\n\n".join(f"$${v}$$" for v in eqs.values()))
    subprocess.run(["pandoc", md, "-o", out], check=True)
    xml = zipfile.ZipFile(out).read("word/document.xml").decode("utf-8")
omml = re.findall(r"<m:oMathPara>.*?</m:oMathPara>", xml, re.S)
assert len(omml) == len(eqs), (len(omml), len(eqs))
json.dump(dict(zip(eqs, omml)), open(os.path.join(HERE, "equations.omml.json"), "w", encoding="utf-8"))
print(f"{len(omml)} equations converted")
