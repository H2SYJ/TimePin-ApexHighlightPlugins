"""Deterministic standard-library tests for the bidirectional JSONL transport."""

from __future__ import annotations

import io
import json
import queue
import threading
import time
import unittest
from contextlib import redirect_stderr

from src.errors import Cancelled, RpcError
from src.protocol import JsonRpcServer


class LineReader:
    def __init__(self) -> None:
        self.lines: queue.Queue[str] = queue.Queue()
        self.read_threads: set[int] = set()

    def readline(self) -> str:
        self.read_threads.add(threading.get_ident())
        return self.lines.get(timeout=5)


class LineWriter:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.lines: list[str] = []
        self.fail = False
        self.flushes = 0

    def write(self, text: str) -> int:
        with self.condition:
            if self.fail:
                raise BrokenPipeError("test output failure")
            self.lines.append(text)
            self.condition.notify_all()
        return len(text)

    def flush(self) -> None:
        self.flushes += 1

    def messages(self) -> list[dict]:
        with self.condition:
            return [json.loads(line) for line in self.lines]

    def wait(self, predicate, timeout: float = 2) -> dict:
        with self.condition:
            if not self.condition.wait_for(
                lambda: any(predicate(json.loads(line)) for line in self.lines), timeout
            ):
                raise AssertionError(f"Expected response not received: {self.lines}")
            return next(
                json.loads(line) for line in self.lines if predicate(json.loads(line))
            )


class Harness:
    def __init__(self, handler, prepare_run=None) -> None:
        self.reader = LineReader()
        self.writer = LineWriter()
        self.server = JsonRpcServer(self.reader, self.writer)
        self.errors: list[Exception] = []

        def serve():
            try:
                self.server.serve(handler, prepare_run=prepare_run)
            except Exception as exc:
                self.errors.append(exc)

        self.thread = threading.Thread(target=serve, name="test-reader", daemon=True)
        self.thread.start()

    def send(self, value) -> None:
        self.reader.lines.put(json.dumps(value) + "\n")

    def reply(self, request_id, timeout: float = 2) -> dict:
        return self.writer.wait(
            lambda message: message.get("id") == request_id and "method" not in message,
            timeout,
        )

    def finish(self) -> None:
        if self.thread.is_alive():
            self.send(rpc("shutdown", "test-cleanup"))
        self.join()

    def join(self) -> None:
        self.thread.join(3)
        if self.thread.is_alive():
            raise AssertionError("Reader did not stop")
        if self.errors:
            raise AssertionError(self.errors)


def rpc(method: str, request_id=1, params=None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {} if params is None else params,
    }


def static_messages(values, handler=None):
    if handler is None:

        def handler(method, params):
            return params

    reader = io.StringIO("".join(json.dumps(value) + "\n" for value in values))
    writer = LineWriter()
    JsonRpcServer(reader, writer).serve(handler)
    return writer.messages()


class ProtocolTests(unittest.TestCase):
    def harness(self, handler, prepare_run=None):
        harness = Harness(handler, prepare_run)
        self.addCleanup(harness.finish)
        return harness

    def test_validation_and_recovery(self):
        cases = [
            ([], -32600),
            (None, -32600),
            ({}, -32600),
            (rpc("initialize", True), -32600),
            (rpc("initialize", 1.5), -32600),
            (rpc("initialize", []), -32600),
            (rpc(23), -32600),
            ({"jsonrpc": "1.0", "id": 1, "method": "initialize"}, -32600),
            (rpc("initialize", params=[]), -32602),
            (rpc("initialize", params=None) | {"params": None}, -32602),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                output = static_messages(
                    [value, rpc("initialize", "ok", {"name": "时签"})]
                )
                self.assertEqual(output[0]["error"]["code"], expected)
                self.assertEqual(output[1]["result"], {"name": "时签"})
        for request_id in (None, 0, -5, "text", 2**100):
            self.assertEqual(
                static_messages([rpc("initialize", request_id)])[0]["id"], request_id
            )

    def test_parse_errors_and_blank_lines(self):
        for invalid in (
            "{bad",
            '{"value":NaN}',
            '{"value":Infinity}',
            '{"value":-Infinity}',
            '{"value":1e999}',
        ):
            writer = LineWriter()
            reader = io.StringIO(
                "\n \t\n" + invalid + "\n" + json.dumps(rpc("initialize", 7)) + "\n"
            )
            JsonRpcServer(reader, writer).serve(lambda method, params: {})
            output = writer.messages()
            self.assertEqual(
                output[0],
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
            )
            self.assertEqual(output[1]["id"], 7)

    def test_notifications_never_reply_even_on_failure(self):
        calls = []

        def handler(method, params):
            calls.append(method)
            raise RpcError(-32601, "Unknown method")

        output = static_messages(
            [
                {"jsonrpc": "2.0", "method": "unknown"},
                {"jsonrpc": "2.0", "method": "initialize", "params": []},
                {"jsonrpc": "wrong", "method": 12},
                {"jsonrpc": "2.0", "method": "shutdown"},
            ],
            handler,
        )
        self.assertEqual(output, [])
        self.assertEqual(calls, ["unknown", "cancel", "shutdown"])

    def test_handler_errors_are_safe_and_do_not_stop_reader(self):
        def handler(method, params):
            if method == "unknown":
                raise RpcError(-32601, "Unknown method", {"method": method})
            if method == "explode":
                raise RuntimeError("private diagnostic")
            return {}

        with redirect_stderr(io.StringIO()) as diagnostics:
            output = static_messages(
                [rpc("unknown", 1), rpc("explode", 2), rpc("initialize", 3)], handler
            )
        self.assertEqual(
            output[0]["error"],
            {
                "code": -32601,
                "message": "Unknown method",
                "data": {"method": "unknown"},
            },
        )
        self.assertEqual(
            output[1]["error"], {"code": -32603, "message": "Internal error"}
        )
        self.assertIn("private diagnostic", diagnostics.getvalue())
        self.assertEqual(output[2]["result"], {})

    def test_non_json_handler_result_becomes_internal_error(self):
        for bad_result in (object(), float("nan"), {"v": float("inf")}):
            with redirect_stderr(io.StringIO()):
                output = static_messages(
                    [rpc("initialize", 1)], lambda method, params: bad_result
                )
            self.assertEqual(output[0]["error"]["code"], -32603)

    def test_shutdown_runs_cancel_once_and_returns_empty_object(self):
        calls = []
        reader_thread = threading.get_ident()

        def handler(method, params):
            calls.append((method, params, threading.get_ident()))
            return {"not": "shutdown result"}

        output = static_messages([rpc("shutdown", 5), rpc("initialize", 6)], handler)
        self.assertEqual(output, [{"jsonrpc": "2.0", "id": 5, "result": {}}])
        self.assertEqual(
            calls, [("cancel", {}, reader_thread), ("shutdown", {}, reader_thread)]
        )

    def test_prepare_reserves_busy_and_cancellation_is_not_lost(self):
        context = {}
        allow_run = threading.Event()
        entered = threading.Event()
        calls = []

        def prepare(params):
            calls.append(("prepare", threading.get_ident()))
            context["event"] = threading.Event()

        def handler(method, params):
            calls.append((method, threading.get_ident()))
            if method == "run":
                entered.set()
                if not allow_run.wait(2):
                    raise RuntimeError("test run stuck")
                if context["event"].is_set():
                    raise Cancelled()
                raise RuntimeError("Cancellation was lost")
            if method == "cancel":
                context["event"].set()
                allow_run.set()
            return {}

        h = self.harness(handler, prepare)
        h.send(rpc("run", 1))
        self.assertTrue(entered.wait(2))
        h.send(rpc("run", 2))
        h.send(rpc("initialize", 3))
        self.assertEqual(h.reply(2)["error"]["code"], -32001)
        self.assertEqual(h.reply(3)["error"]["code"], -32001)
        h.send(rpc("cancel", 4))
        self.assertEqual(h.reply(4)["result"], {})
        self.assertEqual(h.reply(1)["error"]["code"], -32800)
        h.finish()
        main_id = h.thread.ident
        self.assertEqual(h.reader.read_threads, {main_id})
        self.assertEqual(calls[0], ("prepare", main_id))
        self.assertTrue(all(tid == main_id for name, tid in calls if name != "run"))
        self.assertTrue(all(tid != main_id for name, tid in calls if name == "run"))
        self.assertFalse(h.server._worker.is_alive())
        self.assertTrue(h.server._worker.daemon)

    def test_prepare_error_releases_busy_and_does_not_start_handler(self):
        preparations = []
        calls = []

        def prepare(params):
            preparations.append(params)
            if len(preparations) == 1:
                raise RpcError(-32010, "Cannot prepare")

        def handler(method, params):
            calls.append(method)
            return "ran" if method == "run" else {}

        h = self.harness(handler, prepare)
        h.send(rpc("run", 1))
        self.assertEqual(h.reply(1)["error"]["code"], -32010)
        self.assertNotIn("run", calls)
        h.send(rpc("initialize", 2))
        self.assertEqual(h.reply(2)["result"], {})
        h.send(rpc("run", 3))
        self.assertEqual(h.reply(3)["result"], "ran")

    def test_bidirectional_rpc_keeps_reading_and_does_not_swallow_requests(self):
        h = None

        def handler(method, params):
            if method == "run":
                first = h.server.request(
                    "host/first", {"text": "时签"}, threading.Event(), timeout=2
                )
                second = h.server.request(
                    "host/second", {}, threading.Event(), timeout=2
                )
                return [first, second]
            if method == "ping":
                return "pong"
            return {}

        h = self.harness(handler)
        h.send(rpc("run", 1))
        first = h.writer.wait(lambda m: m.get("method") == "host/first")
        host_id = first["id"]
        self.assertIs(type(host_id), int)
        self.assertGreaterEqual(host_id, 1_000_000_000)
        self.assertLessEqual(host_id, (1 << 64) - 1)
        h.send(rpc("ping", host_id))
        self.assertEqual(h.reply(host_id)["result"], "pong")
        h.send({"jsonrpc": "2.0", "id": str(host_id), "result": "must be ignored"})
        h.send({"jsonrpc": "2.0", "id": host_id, "result": "first"})
        second = h.writer.wait(lambda m: m.get("method") == "host/second")
        self.assertEqual(second["id"], host_id + 1)
        h.send({"jsonrpc": "2.0", "id": second["id"], "result": "second"})
        self.assertEqual(h.reply(1)["result"], ["first", "second"])
        self.assertEqual(h.server._pending, {})

    def test_host_error_and_malformed_host_response(self):
        for remote_error in (
            {"code": -32042, "message": "Host failed", "data": {"why": 1}},
            {"code": True, "message": "invalid"},
        ):
            h = None

            def handler(method, params):
                if method == "run":
                    return h.server.request(
                        "host/service", {}, threading.Event(), timeout=2
                    )
                return {}

            h = self.harness(handler)
            h.send(rpc("run", 1))
            host = h.writer.wait(lambda m: m.get("method") == "host/service")
            h.send({"jsonrpc": "2.0", "id": host["id"], "error": remote_error})
            error = h.reply(1)["error"]
            self.assertEqual(
                error["code"], -32603 if remote_error["code"] is True else -32042
            )
            self.assertEqual(h.server._pending, {})
            h.finish()

    def test_external_cancel_and_timeout_clean_pending(self):
        for mode in ("cancel", "timeout"):
            h = None
            cancelled = threading.Event()

            def handler(method, params):
                if method == "run":
                    return h.server.request(
                        "host/wait",
                        {},
                        cancelled,
                        timeout=0.05 if mode == "timeout" else 2,
                    )
                return {}

            h = self.harness(handler)
            h.send(rpc("run", 1))
            host = h.writer.wait(lambda m: m.get("method") == "host/wait")
            if mode == "cancel":
                cancelled.set()
            self.assertEqual(
                h.reply(1)["error"]["code"], -32800 if mode == "cancel" else -32002
            )
            self.assertEqual(h.server._pending, {})
            h.send({"jsonrpc": "2.0", "id": host["id"], "result": "late"})
            h.send(rpc("initialize", 2))
            self.assertEqual(h.reply(2)["result"], {})
            h.finish()

    def test_task_aware_cancel_interrupts_only_the_matching_pending_call(self):
        h = None
        cancel_event = threading.Event()

        def handler(method, params):
            if method == "run":
                return h.server.request("host/wait", {}, cancel_event, timeout=10)
            if method == "cancel":
                matching = not params or params.get("taskId") == "active"
                if matching:
                    cancel_event.set()
                return {"cancelled": matching}
            return {}

        h = self.harness(handler)
        h.send(rpc("run", 1))
        h.writer.wait(lambda m: m.get("method") == "host/wait")
        h.send(rpc("cancel", 2, {"taskId": "wrong"}))
        self.assertEqual(h.reply(2)["result"], {"cancelled": False})
        self.assertFalse(cancel_event.is_set())
        self.assertEqual(len(h.server._pending), 1)
        h.send(rpc("cancel", 3, {"taskId": "active"}))
        self.assertEqual(h.reply(3)["result"], {"cancelled": True})
        self.assertEqual(h.reply(1)["error"]["code"], -32800)
        self.assertEqual(h.server._pending, {})

    def test_eof_cancels_pending_suppresses_worker_response_and_joins(self):
        h = None
        cancelled = threading.Event()

        def handler(method, params):
            if method == "run":
                return h.server.request("host/wait", {}, threading.Event(), timeout=10)
            if method == "cancel":
                cancelled.set()
            return {}

        h = self.harness(handler)
        h.send(rpc("run", 1))
        h.writer.wait(lambda m: m.get("method") == "host/wait")
        h.reader.lines.put("")
        h.join()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(h.server._pending, {})
        self.assertFalse(h.server._worker.is_alive())
        self.assertFalse(
            any("result" in m or "error" in m for m in h.writer.messages())
        )

    def test_shutdown_bounded_join_and_suppresses_late_result_and_notification(self):
        release = threading.Event()
        entered = threading.Event()
        closed = threading.Event()
        h = None

        def handler(method, params):
            if method == "run":
                entered.set()
                release.wait(3)
                try:
                    h.server.notify("late", {})
                except Cancelled:
                    closed.set()
                return "too late"
            return {}

        h = self.harness(handler)
        h.server._JOIN_TIMEOUT = 0.03
        h.send(rpc("run", 1))
        self.assertTrue(entered.wait(2))
        before = time.monotonic()
        with redirect_stderr(io.StringIO()):
            h.send(rpc("shutdown", 2))
            self.assertEqual(h.reply(2)["result"], {})
            h.join()
        self.assertLess(time.monotonic() - before, 1)
        worker = h.server._worker
        self.assertTrue(worker.daemon)
        self.assertTrue(worker.is_alive())
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(closed.is_set())
        self.assertEqual(
            h.writer.messages(), [{"jsonrpc": "2.0", "id": 2, "result": {}}]
        )

    def test_failed_output_wakes_pending_and_stops(self):
        h = None
        cancelled = threading.Event()

        def handler(method, params):
            if method == "run":
                try:
                    return h.server.request(
                        "host/wait", {}, threading.Event(), timeout=10
                    )
                except Cancelled:
                    cancelled.set()
                    raise
            return {}

        h = self.harness(handler)
        h.send(rpc("run", 1))
        h.writer.wait(lambda m: m.get("method") == "host/wait")
        h.writer.fail = True
        with redirect_stderr(io.StringIO()):
            h.send(rpc("initialize", 2))
            h.join()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(h.server._pending, {})
        self.assertFalse(h.server._worker.is_alive())
        with self.assertRaises(Cancelled):
            h.server.notify("progress", {})

    def test_concurrent_notifications_are_complete_utf8_json_lines(self):
        writer = LineWriter()
        server = JsonRpcServer(io.StringIO(), writer)
        threads = [
            threading.Thread(
                target=lambda: [
                    server.notify("progress", {"text": "时签"}) for _ in range(25)
                ]
            )
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2)
        self.assertEqual(len(writer.messages()), 100)
        self.assertEqual(writer.flushes, 100)
        self.assertTrue(
            all(
                "时签" in line and line.endswith("\n") and line.count("\n") == 1
                for line in writer.lines
            )
        )

    def test_outgoing_serialization_failure_cleans_pending(self):
        server = JsonRpcServer(io.StringIO(), LineWriter())
        with self.assertRaises(ValueError):
            server.request("host/service", {"value": float("nan")}, threading.Event())
        self.assertEqual(server._pending, {})
        with self.assertRaises(ValueError):
            server.notify("progress", {"value": float("inf")})

    def test_reader_thread_cannot_deadlock_on_reverse_request(self):
        server = None

        def handler(method, params):
            if method == "initialize":
                return server.request("host/service", {}, threading.Event())
            return {}

        writer = LineWriter()
        server = JsonRpcServer(
            io.StringIO(json.dumps(rpc("initialize")) + "\n"), writer
        )
        server.serve(handler)
        self.assertEqual(writer.messages()[0]["error"]["code"], -32603)
        self.assertEqual(server._pending, {})


if __name__ == "__main__":
    unittest.main()
