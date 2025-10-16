Schema-Guided Dialogue (Banking Subset)
=======================================

This folder now stores only the banking-related slices of the Google DSTC8 Schema-Guided Dialogue corpus. The full dataset was downloaded from the official repository (`google-research-datasets/dstc8-schema-guided-dialogue`). All non-banking domains were removed to reduce the footprint from ~620 MB to ~22 MB.

Layout
------
- `banking_only/`
  - `train/`, `dev/`, `test/` – JSON files containing dialogues whose `services` reference `Banks_*` or `Credit_Cards_*` domains
  - `schema.json` per split – filtered to the same banking services
- `dstc8-schema-guided-dialogue-master/` – documentation assets (license, README, overview graphic, etc.) retained from the original release

Reproducing the pruning
-----------------------
The subset was generated via a Python script that kept any dialogue whose `services` list intersects with the filtered schema (service names containing `Banks` or `Credit_Cards`). The script lives inline in the session history if you need to rerun or modify it.

If you ever require the full multi-domain version again, re-download the upstream zip and expand it beside this directory before running any analytics that expect the additional domains.
