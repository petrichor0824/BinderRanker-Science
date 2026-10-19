# Data field dictionary

## Frozen score tables

Each file under `benchmark/frozen_scores/` is one complete 5,000-candidate target pool. The 132 columns fall into these groups:

- identity/status: `pdb_name`, `error`;
- primary outputs: `final_score_v5`, `rank_final_score_v5`;
- selection diagnostics: `filter_level`, `filter_*_pass`, `filter_*_reasons`, and within-filter ranks;
- coherent-interface terms: `v5_norm_coverage`, `v5_norm_cb_support`, `v5_norm_microfit_raw`, `extent_*_v5`, and `extent_score_v5`;
- morphology terms: `score_line`, `score_plane`, `score_compact`, `morphology_*_score_v5`, and global/local/blended mode weights;
- v4 diagnostics: `final_score_v4`, `score_microfit`, `score_safety`, `score_roughness`, and consensus/component ranks;
- 70 frozen extractor fields: lengths, shell/contact weights, orientation and Cβ support, coverage/entropy/span, spatial-field smoothness, contact-map continuity, PCA morphology, microdomain fit, optional desired/undesired-region summaries, and `error`.

The exact 70-field extractor order is defined by `binderranker.geometry.contact_field.FIELDNAMES`. Formula-level definitions are in `paper/supplementary/Supplementary_Note_1.md`.

## AF2 label tables

- `target_id`, `pdb_id`, `design_index`: candidate identity and target.
- `binder_aligned_rmsd`, `target_aligned_rmsd`: aligned structural deviations.
- `pae_binder`, `pae_interaction`, `pae_target`: PAE summaries.
- `plddt_binder`, `plddt_target`, `plddt_total`: pLDDT summaries.
- `time`: recorded per-candidate processing time in seconds.
- `description`: historical prediction identifier.

Medium and Loose positives are derived computational surrogate labels defined in the paper; they are not wet-lab outcomes.

## Pruning and recall tables

- `retained_fraction` / `pruned_fraction`: complementary target-pool fractions.
- `macro_recall`: unweighted mean recall across the five targets.
- `median_recall`, `min_target_recall`, `max_target_recall`: across-target summaries.
- `target_recall`: required within-target recall for fixed-recall operating points.
- `mean_pruned_fraction`: average removable fraction at that recall requirement.

## Representative structures

`representative_structures_metadata.tsv` maps each PDB to target, selection category, design index, frozen rank/score, AF2 surrogate metrics, Boolean label membership, contact count, and radius of gyration. The original workstation path column is intentionally omitted from the public copy.
