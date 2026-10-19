# BinderRanker v1.0.0

BinderRanker is a training-free, sequence-free method for ranking target-specific pools of protein-binder backbones before sequence design and complex prediction.

## Highlights

- Installable Python package and command-line interface for PDB inspection, descriptor extraction, frozen v5 scoring, ranking, and candidate selection.
- Frozen historical project implementations retained byte-for-byte with SHA256 verification.
- Frozen v5 scoring and ranking parity verified across five held-out pools containing 25,000 candidates.
- Full-pool review covered 1,277,012 numerical cells with no deviation above `1e-12` and no rank, filter, or text mismatch.
- Seventeen release-candidate tests passed, main Figures 1-4 were regenerated successfully, and the Python wheel was built successfully during independent review.
- Benchmark-summary, figure-reproduction, schema-audit, and parity scripts are included.

## Scientific scope

The held-out benchmark used AF2-derived computational surrogate endpoints, not wet-lab binding success. Scores are normalized within one target-specific pool and are not absolute binding probabilities.

`scripts/reproduce_benchmark_summaries.py` summarizes existing aggregate benchmark tables; it does not independently reconstruct the complete benchmark from candidate-level AF2 labels. `scripts/reproduce_figures.py` writes generated output to `figures/reproduced/` by default.

The initial public release contains the MIT-licensed software, tests, historical frozen project sources, and reproducibility scripts. Manuscript files, publication figures, research tables, AF2-derived results, source data, and representative PDBs remain outside the public commit pending separate redistribution approval.

## Citation

Final paper authorship, DOI, and formal citation metadata will be added in a later commit after author confirmation.
