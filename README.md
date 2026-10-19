# BinderRanker

Sequence-free geometric ranking for de novo protein binder design.

BinderRanker ranks a target-specific pool of protein-binder backbones before sequence design and complex prediction. The frozen v5 method is training-free and converts binder-target backbone geometry into a continuous, coherence-aware priority score.

## Scientific motivation

RFdiffusion can generate thousands of target-conditioned backbones, while ProteinMPNN sequence design and AF2/AF3 evaluation add substantial downstream cost. BinderRanker provides an early, high-recall triage step. It does not replace sequence design, structure prediction, molecular dynamics, or experimental validation.

## Algorithm

The frozen v5 model combines:

- an orientation-aware contact field;
- target effective coverage;
- pseudo-Cβ-supported contact;
- local microdomain geometry fit; and
- a target-adaptive morphology prior.

Each interface descriptor is robustly normalized within one complete target-specific candidate pool using its 5th and 95th percentiles. Their geometric mean forms the coherent-interface score. The final ranking is

```text
final_score_v5 = 0.90 * coherent_interface + 0.10 * global_morphology
```

The scientific definitions, operation order, filters, fallback behavior, and tie-breaking follow the byte-protected historical sources in [`reference/frozen_v1`](reference/frozen_v1/).

## Held-out results

The study evaluated five held-out targets containing 25,000 RFdiffusion backbones. At 40% pruning, target-macro recall was approximately 97.5% for the Medium endpoint and 97.8% for the Loose endpoint. At 50% pruning, recall was approximately 94.1% and 93.2%, respectively.

Medium and Loose are AF2-derived computational surrogate endpoints, not wet-lab binding success. BinderRanker scores are relative to one target pool and are not calibrated binding probabilities or a common absolute scale across targets.

## Installation

BinderRanker requires Python 3.10 or newer.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
binderranker version
```

Optional dependencies are available as `.[figures]` and `.[audit]`.

## Quick start

Place the PDB files for one target-specific candidate pool in a directory. Binder chain `A` is the default; all other polymer chains are treated as the target.

```bash
# Check chain assignment and target identity
binderranker inspect --input-dir path/to/one_target_pool --binder-chain A --strict

# Extract frozen geometry descriptors
binderranker extract --input-dir path/to/one_target_pool --binder-chain A --strict --output descriptors.csv

# Score the complete descriptor pool and select the top 20%
binderranker score --input descriptors.csv --output ranked.tsv --top-fraction 0.20

# Extract, score, rank, and select directly from PDB files
binderranker rank --input-dir path/to/one_target_pool --binder-chain A --strict --output ranked.tsv --top-k 100
```

One run must contain candidates for exactly one target. Pool normalization is part of the scientific definition: do not score chunks independently and concatenate them.

The primary output contains every valid candidate sorted by `rank_final_score_v5`. `final_score_v5` determines the continuous within-pool ranking. Broad, medium, and strict filters are diagnostic columns. `--top-k` and `--top-fraction` write a separate selected table after ranking.

## Reproducibility

```bash
python -m unittest discover -s tests -v
python scripts/verify_frozen_sources.py
```

The independently reviewed release candidate passed 17 tests and reproduced frozen v5 scoring and ranking across five 5,000-candidate pools. The full-pool comparison checked 1,277,012 numerical cells with no deviation above `1e-12` and no rank, filter, or text mismatch.

The external parity tests require the separately retained, untracked evidence pack:

```powershell
$env:BINDERRANKER_EVIDENCE_ROOT = 'D:\path\to\submission_evidence_pack_v1_20261015_181940'
python -m unittest discover -s tests -v
```

The repository includes benchmark-summary and figure-reproduction scripts. Their aggregate source tables, publication figures, candidate-level results, AF2-derived labels, representative PDBs, manuscript, and supplement are withheld from the initial public commit pending separate redistribution approval. See [`data/LICENSE_STATUS.md`](data/LICENSE_STATUS.md), [`paper/README.md`](paper/README.md), and [`figures/README.md`](figures/README.md).

`reproduce_benchmark_summaries.py` reads existing aggregate source tables; it does not rebuild the benchmark independently from candidate-level labels. `reproduce_figures.py` writes to `figures/reproduced/` by default and accepts `--output-dir`.

## Citation and license

The software version is `1.0.0`. Final paper authorship, DOI, and citation metadata will be added after author confirmation; no placeholder `CITATION.cff` is included in the public commit.

Project-owned original software is released under the [MIT License](LICENSE). The MIT license does not grant rights to withheld manuscripts, figures, research tables, AF2-derived results, RFdiffusion-derived structures, upstream PDB content, or third-party dependencies. See [`COPYRIGHT_AND_DATA.md`](COPYRIGHT_AND_DATA.md) for the source and scope statement.
