# Editing Jupyter Notebooks

`.ipynb` files are JSON. VS Code may present them in an internal XML representation where standard text edits don't persist.

## Option 1: Use `edit_notebook_file` tool (VS Code)

```python
edit_notebook_file(
    cellId="#VSC-xxxxx",
    editType="edit",
    filePath="/path/to/notebook.ipynb",
    language="python",
    newCode="# updated code"
)
```

## Option 2: Edit raw JSON directly (most reliable)

If edits don't persist, close the notebook in VS Code and edit the JSON:

```bash
sed -i 's/old_text/new_text/g' notebook.ipynb
```

**Key**: If notebook edits aren't persisting, close the notebook and edit the raw JSON file directly.
