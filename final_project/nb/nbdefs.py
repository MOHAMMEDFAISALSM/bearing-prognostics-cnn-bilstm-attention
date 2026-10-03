"""Tiny helpers to assemble the notebook from part files."""
import uuid
CELLS = []


def md(text):
    text = text.strip("\n")
    CELLS.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    text = text.strip("\n")
    CELLS.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)})
