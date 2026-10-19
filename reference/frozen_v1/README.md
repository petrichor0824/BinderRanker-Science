# Frozen BinderRanker v1 scientific sources

The four Python files in this directory are byte-for-byte copies from the v1.2 manuscript archive. They are historical scientific evidence and must not be formatted or edited.

Run `python scripts/verify_frozen_sources.py` from the repository root to verify them against `SHA256SUMS.txt`. The same check is included in the automated test suite.

The historical wrapper retains two `/home/...` defaults and historical dependencies intentionally because this directory is byte-protected by SHA256. Those defaults are not used by the installed package or public CLI; active implementation and reproduction scripts use caller-supplied or repository-relative paths.

The v1.2 top-level manifest names manuscript files as `BinderRanker_Manuscript_Final_for_Teacher_v1.2.{md,docx,pdf}`. Locally they are named `Backbone geometry enables high-recall pruning before sequence.{md,docx,pdf}`; each renamed file has the exact hash expected for its historical name. This is a path mapping, not scientific-content corruption.
