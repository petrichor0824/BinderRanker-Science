# Copyright, source, and data scope

## Software

The repository owner has authorized project-owned original software for release under the MIT License. This includes the active Python package, command-line interface, tests, and project scripts.

The files in `reference/frozen_v1/` are byte-for-byte historical project sources preserved for scientific traceability. Their headers contain no third-party copyright notice. `rank_candidates_v4_layered.py` records that it was generated with ChatGPT for the user's RFdiffusion binder-ranking project. The repository does not present those files as RFdiffusion, ProteinMPNN, AlphaFold, or RCSB PDB source code.

BinderRanker depends on external Python packages such as NumPy and has optional integrations with pandas, Matplotlib, Pillow, and openpyxl. Those packages are not vendored here and remain under their own licenses.

## Research materials

The MIT License does not apply to manuscripts, PDFs, publication figures, candidate-level tables, AF2-derived results, source-data tables, RFdiffusion-derived structures, or upstream target coordinates merely because they are part of the local research project.

Those research materials are excluded from the initial public Git commit. Their local provenance and hashes are retained unchanged. They may be added after the project owner confirms contributor approval, the intended data and figure licenses, and any attribution or redistribution obligations inherited from source structures and upstream tools.
