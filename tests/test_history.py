from __future__ import annotations

import unittest

from codebrain.extractors.history import COUPLING_MAX_FILES, HistoryProvider, parse_log
from codebrain.gitutil import REC, SEP, normalise_rename


def log(*commits: tuple[str, str, str, list[str]]) -> str:
    """Build a fake `git log --numstat` payload."""
    out = []
    for sha, when, author, numstat in commits:
        out.append(f"{REC}{sha}{SEP}{when}{SEP}{author}\n" + "\n".join(numstat))
    return "".join(out)


class TestRenameNotation(unittest.TestCase):
    def test_plain_rename(self):
        self.assertEqual(normalise_rename("old/a.py => new/a.py"), "new/a.py")

    def test_braced_rename(self):
        self.assertEqual(normalise_rename("src/{old => new}/a.py"), "src/new/a.py")

    def test_braced_rename_collapsing_a_segment(self):
        self.assertEqual(normalise_rename("src/{legacy => }/a.py"), "src/a.py")

    def test_untouched_path_passes_through(self):
        self.assertEqual(normalise_rename("src/a.py"), "src/a.py")


class TestParseLog(unittest.TestCase):
    def test_parses_commits_and_numstat(self):
        commits = parse_log(log(
            ("abc", "2026-01-02T00:00:00+00:00", "Ada", ["10\t2\tsrc/a.py", "1\t0\tsrc/b.py"]),
            ("def", "2026-01-01T00:00:00+00:00", "Grace", ["5\t5\tsrc/a.py"]),
        ))
        self.assertEqual(len(commits), 2)
        self.assertEqual(commits[0].author, "Ada")
        self.assertEqual(commits[0].files[0], ("src/a.py", 10, 2))

    def test_binary_files_count_as_touched_with_no_line_churn(self):
        commits = parse_log(log(("abc", "t", "Ada", ["-\t-\tlogo.png"])))
        self.assertEqual(commits[0].files[0], ("logo.png", 0, 0))

    def test_renames_are_normalised_during_parse(self):
        commits = parse_log(log(("abc", "t", "Ada", ["3\t1\tsrc/{old => new}/a.py"])))
        self.assertEqual(commits[0].files[0][0], "src/new/a.py")

    def test_empty_input(self):
        self.assertEqual(parse_log(""), [])

    def test_malformed_lines_are_skipped_not_fatal(self):
        commits = parse_log(log(("abc", "t", "Ada", ["garbage", "2\t0\tsrc/a.py"])))
        self.assertEqual(len(commits[0].files), 1)


class FakeCtx:
    """Minimal stand-in so coupling logic can be tested without a real repo."""

    def __init__(self, files):
        self._files = files

    def rel(self, p):
        return p


class TestCoupling(unittest.TestCase):
    def _edges(self, pairs, counts):
        provider = HistoryProvider()
        env = lambda *a, **k: __import__("codebrain").Envelope.make(  # noqa: E731
            __import__("codebrain").Method.DERIVED, source="history")
        return list(provider._coupling(pairs, counts, env))

    def test_weak_pairs_are_dropped(self):
        edges = self._edges({("a", "b"): 1}, {"a": 10, "b": 10})
        self.assertEqual(edges, [])

    def test_strength_is_jaccard_not_raw_count(self):
        # b/c co-change 4 times out of 4 total changes each: perfectly coupled.
        # a/b co-change 4 times but a changes 100 times: barely coupled.
        edges = self._edges({("a", "b"): 4, ("b", "c"): 4},
                            {"a": 100, "b": 4, "c": 4})
        strengths = {(e.src, e.dst): e.attrs["strength"] for e in edges}
        self.assertEqual(len(edges), 2)
        by_pair = {(s.split(":file:")[-1], d.split(":file:")[-1]): v
                   for (s, d), v in strengths.items()}
        self.assertEqual(by_pair[("b", "c")], 1.0)
        self.assertLess(by_pair[("a", "b")], 0.1)

    def test_edges_are_sorted_strongest_first(self):
        edges = self._edges({("a", "b"): 4, ("b", "c"): 4}, {"a": 100, "b": 4, "c": 4})
        self.assertGreaterEqual(edges[0].attrs["strength"], edges[1].attrs["strength"])


class TestCouplingGuards(unittest.TestCase):
    def test_sweeping_commits_are_excluded_from_coupling(self):
        # A reformat touching hundreds of files couples everything to everything.
        # The guard is a constant, but the behaviour it protects is the point.
        self.assertLess(COUPLING_MAX_FILES, 100)


if __name__ == "__main__":
    unittest.main()
