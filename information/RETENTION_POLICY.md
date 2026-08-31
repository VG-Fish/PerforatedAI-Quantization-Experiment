# Retention policy for generated material

The working tree carries roughly 48 GB, almost none of it versioned. This policy
says what each category is, who may delete it, and what has to exist first. Its
purpose is narrow: make it possible to reclaim space **without** destroying the
only record of a finding.

The inventory this policy operates on is [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md),
regenerated with `uv run dqb evidence_index`.

## Categories

| category | examples | may be deleted | prerequisite |
|---|---|---|---|
| **source** | `src/`, `tests/`, `pyproject.toml`, tracked `experiments/dynamic12/` entrypoints | never, outside normal review | — |
| **reproducible configuration** | run scripts, `.env` template, `sonar-project.properties` | never | — |
| **canonical evidence** | the result tree a current claim rests on | no | must stay until the claim is retired or re-derived from a newer run |
| **historical evidence** | superseded run namespaces (`experiments/dynamic9`–`dynamic11`, `results/top10`, `results/dynamic5`) | yes | indexed in `evidence_index.json` with `--verify`, and any finding still cited has been written into a document under `information/` |
| **disposable cache** | `data/` (~20 GB), `.uv-cache/`, `.ruff_cache/`, `.pytest_cache/`, `.scannerwork/`, `__pycache__/`, `*.egg-info/`, `.DS_Store`, stale PID files | yes | none — `uv run dqb download_data` rebuilds `data/` |
| **archive blobs** | `archive/*.zip` (~15 GB) | yes | contents listed in the evidence index, or confirmed to duplicate a tree that is itself retained |
| **logs** | `logs*/`, worker stream logs | yes | keep the logs of any run whose numbers are still cited; they are the only record of switch/termination behaviour for pre-manifest runs |
| **installed tooling** | `.claude/skills/`, `.perforated_tools/` | yes | regenerable. `PAI Skills/` is the source bundle (it carries `install.sh`/`uninstall.sh`); `.claude/skills/` is its installed copy and was byte-identical at the last check. Keep the bundle, reinstall the copy. |

## Rules

1. **Index before deleting.** Run `uv run dqb evidence_index --verify` and commit the
   updated `information/evidence_index.json` and `EVIDENCE_INDEX.md` in the change that
   deletes a tree. The index is what survives the deletion.
2. **Never delete an unindexed tree**, even one that looks obsolete. As of the last
   index, every stored training record predates the artifact manifest and reports
   `unknown` — none of them can be re-derived by re-reading the files, only by re-running.
3. **`data/` is a cache, not evidence.** It is not versioned, it is not indexed, and it
   is the first thing to delete when space is needed. Dataset identity is carried by each
   artifact's `dataset_revision`, not by the bytes in `data/`.
4. **Deletion is a reviewed change, not a cleanup step.** `uv run dqb clean` removes only
   the paths a previous `dqb` invocation recorded in `.dqb/command_config.json`; it is not
   a tool for retiring historical evidence.
5. **Quarantine before removal.** A tree suspected of being stale or contaminated stays in
   place, marked in the evidence index, until its provenance is captured. The August 30
   audit found stale PAI trees contaminating live runs — the failure mode is silent reuse,
   not disk pressure.
6. **A claim outlives its tree only if it is written down.** Before a historical namespace
   is deleted, the finding it supports must exist as prose in `information/` with its run
   namespace named, so the claim can be traced back to an index entry.

## Current disposition

| tree | size | disposition |
|---|---|---|
| `experiments/dynamic12/` | largest single namespace group | canonical evidence for the current audits; retain |
| `experiments/dynamic9`–`dynamic11` | ~1 GB | historical; deletable once verified-indexed |
| `results/top10`, `results/dynamic5` | ~1.3 GB | historical; superseded by the dynamic12 namespaces |
| `archive/*.zip` | ~15 GB | list contents, then delete unless unique |
| `data/` | ~20 GB | disposable cache; rebuild on demand |
| `logs*/` | ~129 MB | keep the dynamic12 logs; the rest are deletable once indexed |

Nothing in this policy has been executed: no generated tree has been deleted. The
policy plus the evidence index are the prerequisite the audit asked for before any
deletion happens.
