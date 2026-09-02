"""Tests that keep the documentation set honest.

Two failure modes are being prevented.  First, drift: the generated guide
restates the roster, the condition grid, the validity rules and the CLI, and
those must be rendered from the code rather than remembered.  Second, silent
history: a hand-written document that no longer describes the current state has
to say so at the top and be listed in the index, so nothing reads as current
guidance by accident.
"""

import re
import tempfile
import unittest
from pathlib import Path

from dendritic_benchmark.docs import (
    CURRENT_GUIDE_PATH,
    GENERATED_BANNER,
    current_guide_is_current,
    render_current_guide,
    write_current_guide,
)
from dendritic_benchmark.evidence import (
    EVIDENCE_INDEX_MARKDOWN,
    build_evidence_index,
    render_evidence_markdown,
    write_evidence_index,
)
from dendritic_benchmark.model_adapters import DEFAULT_MODEL_KEYS
from dendritic_benchmark.specs import CONDITION_SPECS, MODEL_SPECS
from dendritic_benchmark.statistics import MINIMUM_PAIRED_SEEDS

# Anchored on the repository root so the suite does not depend on the working
# directory pytest happened to be started from.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INFORMATION = _REPOSITORY_ROOT / "information"
_HISTORICAL_INDEX = _INFORMATION / "HISTORICAL_INDEX.md"
_STATUS_BANNER_MARKER = "<!-- status-banner -->"
#: Generated or self-describing documents, which carry no status banner.
_UNBANNERED = {
    "CURRENT_GUIDE.md",
    "EVIDENCE_INDEX.md",
    "HISTORICAL_INDEX.md",
    "RETENTION_POLICY.md",
}


class GeneratedGuideTests(unittest.TestCase):
    def test_the_checked_in_guide_matches_the_registries(self) -> None:
        self.assertTrue(
            current_guide_is_current(_REPOSITORY_ROOT / CURRENT_GUIDE_PATH),
            "information/CURRENT_GUIDE.md is stale; run 'uv run dqb docs'",
        )

    def test_the_guide_is_rendered_from_the_code_not_transcribed(self) -> None:
        text = render_current_guide()
        self.assertIn(GENERATED_BANNER, text)
        for spec in MODEL_SPECS:
            self.assertIn(f"`{spec.key}`", text)
            self.assertIn(spec.dataset, text)
        for condition in CONDITION_SPECS:
            self.assertIn(f"`{condition.key}`", text)
        for model_key in DEFAULT_MODEL_KEYS:
            self.assertIn(model_key, text)
        # The validity contract and the seed rule are quoted from the modules
        # that enforce them, so they cannot say something the code does not do.
        self.assertIn("artifact_manifest.json", text)
        self.assertIn("verified_retained", text)
        self.assertIn(str(MINIMUM_PAIRED_SEEDS), text)
        # Every subcommand the CLI registers is documented.
        for command in (
            "dqb run",
            "dqb download_data",
            "dqb compare",
            "dqb generate_graphs",
            "dqb benchmark_models",
            "dqb clean",
            "dqb docs",
            "dqb evidence_index",
        ):
            self.assertIn(f"### `{command}`", text)

    def test_generation_is_deterministic_and_check_detects_drift(self) -> None:
        self.assertEqual(render_current_guide(), render_current_guide())
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = Path(root) / "guide.md"
            self.assertFalse(current_guide_is_current(path))
            write_current_guide(path)
            self.assertTrue(current_guide_is_current(path))
            path.write_text(path.read_text() + "hand edit\n")
            self.assertFalse(current_guide_is_current(path))


class DocumentationIndexTests(unittest.TestCase):
    def _documents(self) -> list[Path]:
        # The whole tree, not just the top level: the index's own rule 1 says
        # "every document under information/", and the working sets that live
        # in subdirectories (base_examples/, problems/, results_analysis/) are
        # exactly the ones that go stale unnoticed.
        return sorted(_INFORMATION.rglob("*.md"))

    def test_every_hand_written_document_declares_its_status(self) -> None:
        for path in sorted(_INFORMATION.glob("*.md")):
            if path.name in _UNBANNERED:
                continue
            with self.subTest(document=path.name):
                head = path.read_text().split("\n\n", 3)
                self.assertIn(
                    _STATUS_BANNER_MARKER,
                    "\n\n".join(head[:3]),
                    f"{path.name} needs a status banner under its title",
                )

    def test_every_document_is_indexed_with_a_status(self) -> None:
        index = _HISTORICAL_INDEX.read_text()
        for path in self._documents():
            if path == _HISTORICAL_INDEX:
                continue
            with self.subTest(document=path.name):
                self.assertIn(
                    path.name,
                    index,
                    f"{path.name} is not listed in information/HISTORICAL_INDEX.md",
                )
        for status in ("current (generated)", "historical", "superseded"):
            self.assertIn(status, index)

    def test_superseded_documents_name_their_replacement(self) -> None:
        """Every Superseded row names a replacement, and it is a real file.

        Asserted as an invariant rather than against a fixed pair of filenames:
        a superseded document is eventually deleted, and a test that named one
        of them started failing the day that happened while saying nothing
        about whether the rule still held.
        """
        index = _HISTORICAL_INDEX.read_text()
        superseded_section = index.split("## Superseded", 1)[1]
        rows = [
            line
            for line in superseded_section.splitlines()
            if line.startswith("| [")
        ]
        self.assertTrue(rows, "the Superseded section has no rows")
        for row in rows:
            with self.subTest(row=row):
                targets = re.findall(r"\]\(([^)]+\.md)\)", row)
                self.assertGreaterEqual(
                    len(targets), 2, "a superseded row must name a replacement"
                )
                for target in targets:
                    self.assertTrue(
                        (_INFORMATION / target).exists(),
                        f"{target} is linked from the index but does not exist",
                    )

    def test_the_index_links_only_to_documents_that_exist(self) -> None:
        index = _HISTORICAL_INDEX.read_text()
        for target in sorted(set(re.findall(r"\]\(([^)]+\.md)\)", index))):
            with self.subTest(link=target):
                self.assertTrue(
                    (_INFORMATION / target).exists(),
                    f"information/HISTORICAL_INDEX.md links to a missing {target}",
                )

    def test_the_retention_policy_is_indexed_and_names_the_evidence_index(self) -> None:
        policy = (_INFORMATION / "RETENTION_POLICY.md").read_text()
        self.assertIn("EVIDENCE_INDEX.md", policy)
        self.assertIn("dqb evidence_index", policy)
        # data/ is a cache, never evidence: deleting it must stay cheap.
        self.assertIn("`data/`", policy)


class EvidenceIndexTests(unittest.TestCase):
    def test_indexing_a_tree_records_provenance_without_hashing_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            condition_dir = root_path / "run" / "lenet5" / "dendrites_fp32"
            condition_dir.mkdir(parents=True)
            (condition_dir / "record.json").write_text(
                '{"model_key": "lenet5", "condition_key": "dendrites_fp32", '
                '"artifact_id": "abc", "metric_name": "Accuracy", '
                '"metric_value": 0.91, "dendrite_audit_status": "legacy_unchecked"}'
            )
            index = build_evidence_index([root_path], verify=False)
            self.assertEqual(len(index.artifacts), 1)
            entry = index.artifacts[0]
            self.assertEqual(entry.model_key, "lenet5")
            self.assertEqual(entry.condition_key, "dendrites_fp32")
            self.assertEqual(entry.dendrite_audit_status, "legacy_unchecked")
            # No manifest at all: the tree predates artifact identity and must
            # not be presented as verified.
            self.assertEqual(entry.manifest_status, "unknown")
            self.assertEqual(entry.manifest_state, "missing")
            self.assertEqual(entry.run_namespace, str(root_path / "run"))

            markdown = render_evidence_markdown(index)
            self.assertIn("# Evidence index", markdown)
            self.assertIn("RETENTION_POLICY.md", markdown)
            self.assertIn("`unknown`", markdown)

    def test_a_missing_root_is_reported_rather_than_skipped(self) -> None:
        index = build_evidence_index(["definitely-not-a-directory"], verify=False)
        self.assertEqual(len(index.roots), 1)
        self.assertFalse(index.roots[0].exists)
        self.assertEqual(index.artifacts, [])

    def test_the_index_round_trips_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            index = build_evidence_index([root_path], verify=False)
            json_path, markdown_path = write_evidence_index(
                index,
                json_path=root_path / "index.json",
                markdown_path=root_path / "INDEX.md",
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertIn("schema_version", json_path.read_text())

    def test_the_checked_in_evidence_index_exists(self) -> None:
        self.assertTrue(
            (_REPOSITORY_ROOT / EVIDENCE_INDEX_MARKDOWN).is_file(),
            "run 'uv run dqb evidence_index' and commit the result",
        )


if __name__ == "__main__":
    unittest.main()
