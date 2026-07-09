# Working with Notebooks

We track `.ipynb` files alongside paired `.py` files for readable PR diffs. CI enforces they stay in sync.

## Setup (one-time, optional but recommended)

```bash
pip install jupytext pre-commit
pre-commit install
```

If you install pre-commit, syncing happens automatically on commit. If you don't, you'll need to sync manually (see below).

## Daily workflow

1. Edit the `.ipynb` as usual
2. Before committing, sync the paired `.py`:
   ```bash
   jupytext --sync *.ipynb
   ```
3. Commit both files:
   ```bash
   git add *.ipynb *.py
   git commit -m "your message"
   ```

## What happens in PRs

- `.ipynb` shows "Binary file changed" (no noisy JSON diff)
- `.py` shows the actual code diff — **this is what you review**
- CI checks that `.py` matches `.ipynb`. If you forgot to sync, the PR is blocked with instructions

## If CI fails

The error message will tell you exactly what to do:

```bash
jupytext --sync *.ipynb
git add *.py
git commit -m "Sync .py with notebook changes"
git push
```

## Adding a new notebook

Just create your `.ipynb` and run `jupytext --sync your_notebook.ipynb`. Commit both files.
