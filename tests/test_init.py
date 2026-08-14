from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from codebrain import cli


def init(root: Path, *extra: str) -> str:
    out = io.StringIO()
    with redirect_stdout(out):
        cli.main(["init", str(root), *extra])
    return out.getvalue()


class TestInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_writes_all_three_integration_points(self):
        init(self.root)
        self.assertTrue((self.root / ".brain").is_dir())
        self.assertTrue((self.root / ".mcp.json").is_file())
        self.assertTrue((self.root / ".claude" / "settings.json").is_file())
        self.assertTrue((self.root / "CLAUDE.md").is_file())

    def test_mcp_config_shape(self):
        init(self.root)
        config = json.loads((self.root / ".mcp.json").read_text(encoding="utf-8"))
        self.assertEqual(config["mcpServers"]["codebrain"]["command"], "codebrain")

    def test_all_five_hook_events_are_wired(self):
        init(self.root)
        hooks = json.loads((self.root / ".claude" / "settings.json")
                           .read_text(encoding="utf-8"))["hooks"]
        for event in ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse"):
            self.assertIn(event, hooks)

    def test_existing_mcp_servers_are_preserved(self):
        (self.root / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"other": {"command": "x"}}}), encoding="utf-8")
        init(self.root)
        config = json.loads((self.root / ".mcp.json").read_text(encoding="utf-8"))
        self.assertIn("other", config["mcpServers"])
        self.assertIn("codebrain", config["mcpServers"])

    def test_existing_hooks_are_preserved(self):
        settings = self.root / ".claude" / "settings.json"
        settings.parent.mkdir()
        settings.write_text(json.dumps({
            "hooks": {"PreToolUse": [{"matcher": "Bash",
                                      "hooks": [{"type": "command",
                                                 "command": "my-linter"}]}]},
            "env": {"KEEP": "me"},
        }), encoding="utf-8")
        init(self.root)
        data = json.loads(settings.read_text(encoding="utf-8"))
        self.assertEqual(data["env"]["KEEP"], "me")
        commands = json.dumps(data["hooks"]["PreToolUse"])
        self.assertIn("my-linter", commands)
        self.assertIn("codebrain guard", commands)

    def test_running_twice_does_not_duplicate(self):
        init(self.root)
        init(self.root)
        hooks = json.loads((self.root / ".claude" / "settings.json")
                           .read_text(encoding="utf-8"))["hooks"]
        self.assertEqual(json.dumps(hooks).count("codebrain guard"), 1)
        self.assertEqual(
            (self.root / "CLAUDE.md").read_text(encoding="utf-8")
            .count("codebrain:start"), 1)

    def test_existing_claude_md_is_appended_to_not_replaced(self):
        (self.root / "CLAUDE.md").write_text("# House rules\n\nUse tabs.\n",
                                             encoding="utf-8")
        init(self.root)
        text = (self.root / "CLAUDE.md").read_text(encoding="utf-8")
        self.assertIn("House rules", text)
        self.assertIn("This repository has a Brain", text)

    def test_unreadable_config_is_left_alone_rather_than_clobbered(self):
        (self.root / ".mcp.json").write_text("{not json", encoding="utf-8")
        output = init(self.root)
        self.assertIn("left alone", output)
        self.assertEqual((self.root / ".mcp.json").read_text(encoding="utf-8"),
                         "{not json")

    def test_opt_outs(self):
        init(self.root, "--no-mcp", "--no-hooks", "--no-claude-md")
        self.assertFalse((self.root / ".mcp.json").exists())
        self.assertFalse((self.root / ".claude").exists())
        self.assertFalse((self.root / "CLAUDE.md").exists())
        self.assertTrue((self.root / ".brain").is_dir())

    def test_missing_directory_is_an_error(self):
        self.assertEqual(cli.main(["init", str(self.root / "nope")]), 2)


if __name__ == "__main__":
    unittest.main()
