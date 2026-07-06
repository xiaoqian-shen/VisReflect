"""HuggingFace-backed data staging.

Benchmarks and checkpoints are read straight from HuggingFace by repo id (e.g.
``craigwu/vstar_bench``). Every resolver ALSO accepts an existing local directory / file
and returns it unchanged, so pre-downloaded data and local checkpoints work with no network
access. Downloads are cached by ``huggingface_hub`` under ``HF_HOME``.
"""

import os

from huggingface_hub import hf_hub_download, snapshot_download


def resolve_repo(source: str, allow_patterns=None) -> str:
    """Local directory for a *dataset* ``source``.

    If ``source`` is an existing local dir it is returned as-is; otherwise it is treated as a
    HuggingFace dataset repo id and ``snapshot_download``-ed (returns the local snapshot dir).
    ``allow_patterns`` restricts which files are fetched (e.g. only the parquet)."""
    if os.path.isdir(source):
        return source
    return snapshot_download(
        repo_id=source, repo_type="dataset", allow_patterns=allow_patterns
    )


def resolve_file(source: str, filename: str) -> str:
    """Local path to ``filename`` within a *dataset* ``source``.

    Joins directly when ``source`` is a local dir; otherwise ``hf_hub_download``s the single
    file from the dataset repo (cheaper than snapshotting the whole repo)."""
    if os.path.isdir(source):
        return os.path.join(source, filename)
    return hf_hub_download(repo_id=source, repo_type="dataset", filename=filename)


def resolve_model(source: str) -> str:
    """Local directory for a *model* ``source`` (a checkpoint dir or a HuggingFace model id).

    Local dirs pass through unchanged; a model repo id is ``snapshot_download``-ed so the
    returned path can be handed straight to ``from_pretrained``."""
    if os.path.isdir(source):
        return source
    return snapshot_download(repo_id=source, repo_type="model")
