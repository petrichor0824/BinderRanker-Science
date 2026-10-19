# BinderRanker data

The complete local release candidate contains five 5,000-row frozen-v5 score tables, corresponding AF2-derived labels, benchmark and runtime summaries, figure source tables, and 21 representative normalized RFdiffusion designs. These materials were validated against a 70-entry SHA256 manifest.

The initial public Git commit includes the schema documentation but withholds the research tables and structures until their separate redistribution terms are approved. See [`LICENSE_STATUS.md`](LICENSE_STATUS.md) and [`FIELD_DICTIONARY.md`](FIELD_DICTIONARY.md).

When authorized data are added, each 5,000-row target table must be normalized as one complete pool. The tables must never be split into independently normalized chunks. AF2 labels are computational surrogate endpoints, not experimental binding measurements, and the 21 representative PDBs support descriptor parity only.
