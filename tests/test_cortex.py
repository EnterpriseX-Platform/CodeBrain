from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codebrain.cortex import (
    CONFIG_FILE,
    cross_repo_impact,
    load_cortex,
    parse_config,
    render_impact,
    render_roster,
)
from codebrain.providers import BuildContext, build
from codebrain.store import save
from codebrain.extractors.behavior import BehaviorProvider
from codebrain.extractors.census import CensusProvider
from codebrain.extractors.constraints import ConstraintsProvider


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        write(root, rel, text)
    brain = build(BuildContext(root=root),
                  [CensusProvider(), BehaviorProvider(), ConstraintsProvider()]).brain
    save(brain, root / ".brain")


def two_repo_cortex(tmp: str) -> Path:
    """payments-api exposes POST /v1/charges; billing-worker calls it."""
    base = Path(tmp)
    make_repo(base / "payments-api", {
        "api.py": "@app.post('/v1/charges')\ndef charge():\n    return 1\n",
    })
    make_repo(base / "billing-worker", {
        "worker.py":
            "import requests\n"
            "def run():\n"
            "    requests.post(BASE + '/v1/charges', json={})\n",
    })
    make_repo(base / "unrelated-service", {
        "main.py": "def unrelated():\n    return 42\n",
    })

    config = base / CONFIG_FILE
    config.write_text(
        '[[member]]\nname = "payments"\nroot = "payments-api"\n\n'
        '[[member]]\nname = "billing"\nroot = "billing-worker"\n\n'
        '[[member]]\nname = "unrelated"\nroot = "unrelated-service"\n',
        encoding="utf-8",
    )
    return config


class TestConfigParsing(unittest.TestCase):
    def test_parses_members(self):
        entries = parse_config('[[member]]\nname = "a"\nroot = "../a"\n')
        self.assertEqual(entries, [{"name": "a", "root": "../a", "brain": ""}])

    def test_explicit_brain_path(self):
        entries = parse_config(
            '[[member]]\nname = "a"\nroot = "../a"\nbrain = "../a/out/.brain"\n')
        self.assertEqual(entries[0]["brain"], "../a/out/.brain")

    def test_malformed_toml_yields_no_members_not_an_exception(self):
        self.assertEqual(parse_config("this is not toml [[["), [])

    def test_entries_missing_required_fields_are_skipped(self):
        self.assertEqual(parse_config('[[member]]\nname = "a"\n'), [])

    def test_no_member_table_is_empty(self):
        self.assertEqual(parse_config('title = "x"\n'), [])


class TestLoadCortex(unittest.TestCase):
    def test_missing_config_is_an_empty_roster_not_an_error(self):
        cortex = load_cortex(Path("/definitely/not/here") / CONFIG_FILE)
        self.assertEqual(cortex.members, [])

    def test_loads_every_member(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
        self.assertEqual(len(cortex.loaded), 3)
        self.assertEqual(cortex.failed, [])

    def test_a_member_with_no_brain_is_kept_with_its_error_not_dropped(self):
        # A Cortex that silently drops a broken member gives a confidently
        # incomplete answer, which is worse than one that admits a gap.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "no-brain-here").mkdir()
            config = base / CONFIG_FILE
            config.write_text('[[member]]\nname = "ghost"\nroot = "no-brain-here"\n',
                              encoding="utf-8")
            cortex = load_cortex(config)
        self.assertEqual(len(cortex.members), 1)
        self.assertFalse(cortex.members[0].loaded)
        self.assertTrue(cortex.members[0].error)
        self.assertEqual(cortex.failed, cortex.members)

    def test_never_mutates_a_member_brain_on_disk(self):
        # The one rule that is not negotiable: composed, never re-extracted.
        with tempfile.TemporaryDirectory() as tmp:
            config = two_repo_cortex(tmp)
            manifest = config.parent / "payments-api" / ".brain" / "manifest.json"
            before = manifest.read_bytes()
            load_cortex(config)
            cross_repo_impact(load_cortex(config), "POST /v1/charges")
            self.assertEqual(manifest.read_bytes(), before)


class TestCrossRepoImpact(unittest.TestCase):
    def test_finds_the_route_and_its_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "POST /v1/charges")

        self.assertEqual(report.origin_member, "payments")
        self.assertEqual(report.origin_kind, "route")
        hits = report.by_member()
        self.assertIn("billing", hits)
        self.assertEqual(hits["billing"][0].path, "worker.py")

    def test_the_origin_repo_is_not_reported_as_its_own_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "POST /v1/charges")
        self.assertNotIn("payments", report.by_member())

    def test_an_unrelated_member_reports_no_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "POST /v1/charges")
        self.assertNotIn("unrelated", report.by_member())
        self.assertIn("unrelated", report.scanned)

    def test_only_the_path_is_searched_not_the_http_verb(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "POST /v1/charges")
        self.assertEqual(report.needle, "/v1/charges")

    def test_known_consumers_come_from_the_origins_own_l6_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "POST /v1/charges")
        # A single in-repo route has no L6 public_contract (needs 2+ callers
        # inside the same repo) — this just proves the field wires through
        # without inventing a number when there is genuinely none.
        self.assertIsNone(report.known_consumers)

    def test_a_target_matching_nothing_falls_back_to_a_free_text_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            cortex = load_cortex(two_repo_cortex(tmp))
            report = cross_repo_impact(cortex, "/v1/charges")
        self.assertIsNone(report.origin_member)
        self.assertEqual(report.origin_kind, "unknown")
        self.assertIn("billing", report.by_member())

    def test_an_unreachable_member_is_reported_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            make_repo(base / "payments-api", {
                "api.py": "@app.post('/v1/charges')\ndef charge():\n    return 1\n"})
            config = base / CONFIG_FILE
            config.write_text(
                '[[member]]\nname = "payments"\nroot = "payments-api"\n\n'
                '[[member]]\nname = "ghost"\nroot = "nowhere"\n', encoding="utf-8")
            report = cross_repo_impact(load_cortex(config), "POST /v1/charges")
        self.assertIn("ghost", report.unreachable)

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = two_repo_cortex(tmp)
            first = cross_repo_impact(load_cortex(config), "POST /v1/charges")
            second = cross_repo_impact(load_cortex(config), "POST /v1/charges")
        self.assertEqual([h.path for h in first.hits], [h.path for h in second.hits])


class TestRendering(unittest.TestCase):
    def test_empty_roster_explains_the_config_format(self):
        from codebrain.cortex import Cortex

        text = render_roster(Cortex(config_path=Path(CONFIG_FILE)))
        self.assertIn(CONFIG_FILE, text)
        self.assertIn("[[member]]", text)

    def test_roster_shows_loaded_and_failed_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = render_roster(load_cortex(two_repo_cortex(tmp)))
        self.assertIn("payments", text)
        self.assertIn("billing", text)
        self.assertIn("route(s)", text)

    def test_impact_render_states_it_is_textual_not_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = cross_repo_impact(load_cortex(two_repo_cortex(tmp)),
                                       "POST /v1/charges")
        text = render_impact(report)
        self.assertIn("not proof of a real dependency", text)
        self.assertIn("worker.py:", text)

    def test_output_is_pure_ascii(self):
        # cli.py already had one console-encoding crash this session (P0) that
        # only showed up on a legacy Windows terminal. Cortex output must not
        # reintroduce it by hardcoding a raw unicode dash or checkmark.
        with tempfile.TemporaryDirectory() as tmp:
            config = two_repo_cortex(tmp)
            roster_text = render_roster(load_cortex(config))
            impact_text = render_impact(cross_repo_impact(load_cortex(config),
                                                           "POST /v1/charges"))
        for text in (roster_text, impact_text):
            text.encode("ascii")  # raises UnicodeEncodeError on any non-ASCII char

    def test_no_hits_says_so_plainly(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = cross_repo_impact(load_cortex(two_repo_cortex(tmp)),
                                       "GET /nonexistent")
        self.assertIn("No reference to", render_impact(report))


if __name__ == "__main__":
    unittest.main()
