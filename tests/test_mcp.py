from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from codebrain.mcp_server import PROTOCOL_VERSION, TOOLS, Server, serve
from codebrain.providers import BuildContext, build
from codebrain.store import save
from codebrain.extractors.census import CensusProvider
from codebrain.extractors.operations import OperationsProvider
from codebrain.extractors.structure_py import PythonStructureProvider


def fixture_repo(tmp: str) -> Path:
    root = Path(tmp)
    (root / "pay").mkdir()
    (root / "pay" / "api.py").write_text(
        "def charge_endpoint():\n    return 1\n\n"
        "def refund_endpoint():\n    charge_endpoint()\n", encoding="utf-8")
    (root / "Makefile").write_text("test:\n\tpytest -q\n", encoding="utf-8")
    brain = build(BuildContext(root=root),
                  [CensusProvider(), PythonStructureProvider(),
                   OperationsProvider()]).brain
    save(brain, root / ".brain")
    return root


def rpc(*messages: dict) -> list[dict]:
    stdin = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    stdout = io.StringIO()
    return stdin, stdout


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = fixture_repo(self.tmp.name)
        self.server = Server(str(self.root / ".brain"), str(self.root))

    def test_initialize(self):
        reply = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        result = reply["result"]
        self.assertEqual(result["protocolVersion"], PROTOCOL_VERSION)
        self.assertEqual(result["serverInfo"]["name"], "codebrain")
        self.assertIn("tools", result["capabilities"])

    def test_initialize_instructs_the_agent_to_pack_first(self):
        result = self.server.handle({"jsonrpc": "2.0", "id": 1,
                                     "method": "initialize"})["result"]
        self.assertIn("brain_pack", result["instructions"])

    def test_notifications_get_no_reply(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0",
                                              "method": "notifications/initialized"}))

    def test_tools_list(self):
        reply = self.server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in reply["result"]["tools"]]
        self.assertIn("brain_pack", names)
        self.assertEqual(len(names), len(TOOLS))

    def test_every_tool_declares_a_schema(self):
        for tool in TOOLS:
            self.assertIn("inputSchema", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])

    def test_every_declared_tool_has_a_handler(self):
        handlers = self.server.handlers()
        for tool in TOOLS:
            self.assertIn(tool["name"], handlers)

    def test_unknown_method_is_an_error_not_a_crash(self):
        reply = self.server.handle({"jsonrpc": "2.0", "id": 9, "method": "nope"})
        self.assertEqual(reply["error"]["code"], -32601)

    def test_unknown_notification_is_ignored(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "nope"}))


class TestTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = fixture_repo(self.tmp.name)
        self.server = Server(str(self.root / ".brain"), str(self.root))

    def call(self, name, **args):
        return self.server.call_tool(name, args)

    def test_pack(self):
        text, error = self.call("brain_pack", task="fix charge_endpoint")
        self.assertFalse(error)
        self.assertIn("CONTEXT PACK", text)
        self.assertIn("charge_endpoint", text)

    def test_locate(self):
        text, error = self.call("brain_locate", query="charge_endpoint")
        self.assertFalse(error)
        self.assertIn("charge_endpoint", text)

    def test_locate_miss_says_so(self):
        text, _ = self.call("brain_locate", query="zzzznotathing")
        self.assertIn("Nothing in the Brain", text)

    def test_explain(self):
        text, error = self.call("brain_explain", target="charge_endpoint")
        self.assertFalse(error)
        self.assertIn("provenance", text)
        self.assertIn("evidence", text)

    def test_impact(self):
        text, error = self.call("brain_impact", target="charge_endpoint")
        self.assertFalse(error)
        self.assertIn("refund_endpoint", text)

    def test_impact_with_no_dependents_warns_about_coverage(self):
        text, _ = self.call("brain_impact", target="refund_endpoint")
        self.assertIn("UNKNOWNS", text)

    def test_runbook(self):
        text, error = self.call("brain_runbook")
        self.assertFalse(error)
        self.assertIn("make test", text)

    def test_constraints_absence_is_not_a_guarantee(self):
        text, _ = self.call("brain_constraints", path="pay/api.py")
        self.assertIn("not a guarantee", text)

    def test_unknown_tool(self):
        text, error = self.call("nope")
        self.assertTrue(error)
        self.assertIn("Unknown tool", text)

    def test_a_raising_tool_returns_text_not_an_exception(self):
        # The transport must survive a broken handler, or the session loses the
        # Brain entirely.
        def boom(args, brain):
            raise RuntimeError("kaboom")

        self.server.handlers = lambda: {"brain_pack": boom}
        text, error = self.call("brain_pack", task="x")
        self.assertTrue(error)
        self.assertIn("kaboom", text)


class TestMissingBrain(unittest.TestCase):
    def test_tools_explain_how_to_fix_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = Server(str(Path(tmp) / "nope"), tmp)
            text, error = server.call_tool("brain_runbook", {})
        self.assertTrue(error)
        self.assertIn("codebrain build", text)


class TestTransport(unittest.TestCase):
    def test_end_to_end_over_stdio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_repo(tmp)
            stdin, stdout = rpc(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            serve(str(root / ".brain"), str(root), stdin=stdin, stdout=stdout)
            replies = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln]

        # Two requests, one notification — notifications must not get a reply.
        self.assertEqual([r["id"] for r in replies], [1, 2])

    def test_malformed_json_gets_a_parse_error_and_the_loop_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture_repo(tmp)
            stdin = io.StringIO('{not json\n{"jsonrpc":"2.0","id":5,"method":"ping"}\n')
            stdout = io.StringIO()
            serve(str(root / ".brain"), str(root), stdin=stdin, stdout=stdout)
            replies = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln]

        self.assertEqual(replies[0]["error"]["code"], -32700)
        self.assertEqual(replies[1]["id"], 5)


if __name__ == "__main__":
    unittest.main()
