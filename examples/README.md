# Examples

Run BinderRanker on a directory containing one target-specific pool of PDB files:

```bash
binderranker inspect --input-dir path/to/one_target_pool --binder-chain A --strict
binderranker rank --input-dir path/to/one_target_pool --binder-chain A --strict --output ranked.tsv --top-fraction 0.20
```

The representative RFdiffusion-derived PDB example used during release review is retained locally but is not included in the initial public commit pending redistribution approval. Scores are relative to the complete candidate pool supplied in one run.
