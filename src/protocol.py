"""JSON-RPC 2.0 over UTF-8 text streams, one JSON object per line.

``serve`` owns the only reader. Only ``run`` is dispatched to a daemon worker;
all other handlers (and ``prepare_run``) execute on the reader thread. Callers
must supply UTF-8 TextIO streams and cooperatively stop work on cancellation.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TextIO

from .errors import Cancelled, RpcError

Handler = Callable[[str, dict[str, object]], object]
PrepareRun = Callable[[dict[str, object]], None]


@dataclass
class _Pending:
    ready: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: Exception | None = None


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


def _reject_constant(value: str) -> object:
    raise ValueError(f"Invalid JSON constant: {value}")


def _valid_id(value: object) -> bool:
    return value is None or type(value) in (int, str)


class JsonRpcServer:
    """Bidirectional transport; a server instance can be served only once."""

    _JOIN_TIMEOUT = 1.0
    _CANCEL_POLL = 0.05
    _MAX_ID = (1 << 64) - 1

    def __init__(self, reader: TextIO, writer: TextIO) -> None:
        self._reader = reader
        self._writer = writer
        self._lock = threading.RLock()
        # Always acquire the write lock before the state lock when using both.
        self._write_lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._next_id = 1_000_000_000
        self._run_cancel = threading.Event()
        self._worker: threading.Thread | None = None
        self._reader_thread: threading.Thread | None = None
        self._busy = False
        self._started = False
        self._closing = False
        self._failed = False
        self._cleanup_started = False

    def serve(self, handler: Handler, *, prepare_run: PrepareRun | None = None) -> None:
        """Read until EOF/shutdown, cancelling and boundedly joining the worker.

        ``prepare_run`` reserves application state on the reader thread after
        the transport is busy but before the worker starts. The run handler
        must use that prepared context without clearing its cancellation event.
        """
        with self._lock:
            if self._started:
                raise RuntimeError("JsonRpcServer.serve can only be called once")
            self._started = True
            self._reader_thread = threading.current_thread()
        try:
            while not self._is_closing():
                try:
                    line = self._reader.readline()
                except UnicodeError:
                    self._reply(None, error=RpcError(-32700, "Parse error"))
                    continue
                except Exception as exc:
                    self._diagnostic("Input failed", exc)
                    break
                if self._is_closing() or not line:
                    break
                if not line.strip():
                    continue
                try:
                    message = json.loads(
                        line, parse_constant=_reject_constant, parse_float=_finite_float
                    )
                except (ValueError, RecursionError):
                    self._reply(None, error=RpcError(-32700, "Parse error"))
                    continue
                self._dispatch(message, handler, prepare_run)
        finally:
            self._stop(handler)

    def notify(self, method: str, params: dict[str, object]) -> None:
        """Send an atomic, flushed notification, or raise Cancelled if closed."""
        self._validate_outgoing(method, params)
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(
        self,
        method: str,
        params: dict[str, object],
        cancel_event: threading.Event,
        timeout: float = 330.0,
    ) -> object:
        """Call a host service from a worker while serve continues reading.

        Remote errors are raised as RpcError; cancellation/closure as Cancelled;
        a missing response becomes RpcError(-32002). Pending entries are always
        removed, including when serialization or output fails.
        """
        self._validate_outgoing(method, params)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        if threading.current_thread() is self._reader_thread:
            raise RpcError(-32603, "Host requests cannot block the reader thread")
        pending = _Pending()
        with self._lock:
            run_cancel = self._run_cancel
            if self._closing or cancel_event.is_set() or run_cancel.is_set():
                raise Cancelled("Request cancelled")
            if self._next_id > self._MAX_ID:
                raise RpcError(-32603, "Host request ID space exhausted")
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = pending
        try:
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                cancel_events=(cancel_event, run_cancel),
            )
            deadline = time.monotonic() + timeout
            while True:
                if cancel_event.is_set() or run_cancel.is_set():
                    raise Cancelled("Request cancelled")
                if pending.ready.is_set():
                    if pending.error is not None:
                        raise pending.error
                    return pending.result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RpcError(-32002, "Host request timed out")
                pending.ready.wait(min(remaining, self._CANCEL_POLL))
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def _dispatch(
        self, message: object, handler: Handler, prepare_run: PrepareRun | None
    ) -> None:
        if not isinstance(message, dict):
            self._reply(None, error=RpcError(-32600, "Invalid Request"))
            return
        # A request with an ID equal to a pending host call is still a request.
        if "method" not in message and ("result" in message or "error" in message):
            self._receive_response(message)
            return
        notification = "method" in message and "id" not in message
        request_id = message.get("id")
        valid_id = _valid_id(request_id)
        method = message.get("method")
        if (
            message.get("jsonrpc") != "2.0"
            or not isinstance(method, str)
            or not valid_id
        ):
            if not notification:
                self._reply(
                    request_id if valid_id else None,
                    error=RpcError(-32600, "Invalid Request"),
                )
            return
        params = message.get("params", {})
        if not isinstance(params, dict):
            if not notification:
                self._reply(request_id, error=RpcError(-32602, "Invalid params"))
            return
        if method in ("run", "initialize"):
            with self._lock:
                busy = self._busy
                if method == "run" and not busy:
                    self._busy = True
                    self._run_cancel = threading.Event()
            if busy:
                if not notification:
                    self._reply(
                        request_id, error=RpcError(-32001, "A run is already active")
                    )
                return
        if method == "run":
            try:
                if prepare_run is not None:
                    prepare_run(params)
                worker = threading.Thread(
                    target=self._run,
                    args=(handler, params, request_id, notification),
                    name="jsonrpc-run",
                    daemon=True,
                )
                with self._lock:
                    self._worker = worker
                worker.start()
            except Exception as exc:
                with self._lock:
                    self._worker = None
                    self._busy = False
                if not notification:
                    self._reply(request_id, error=self._as_rpc_error(exc))
        elif method == "shutdown":
            self._stop(handler)
            _, error = self._invoke(handler, method, params)
            if not notification:
                self._reply(request_id, result={}, error=error, allow_closing=True)
        else:
            # Dispatch first: cancel is task-aware. A matching run cancellation sets
            # the task event, which also wakes pending host-service requests.
            result, error = self._invoke(handler, method, params)
            if not notification:
                self._reply(request_id, result=result, error=error)

    def _run(
        self,
        handler: Handler,
        params: dict[str, object],
        request_id: object,
        notification: bool,
    ) -> None:
        try:
            result, error = self._invoke(handler, "run", params)
        finally:
            # Publish availability before the response: once the host observes
            # completion, an immediately following run must not see stale busy.
            with self._lock:
                self._busy = False
        if not notification:
            self._reply(request_id, result=result, error=error)

    def _invoke(
        self, handler: Handler, method: str, params: dict[str, object]
    ) -> tuple[object, RpcError | None]:
        try:
            return handler(method, params), None
        except Exception as exc:
            return None, self._as_rpc_error(exc)

    def _as_rpc_error(self, exc: Exception) -> RpcError:
        if isinstance(exc, RpcError):
            return exc
        if isinstance(exc, Cancelled):
            return RpcError(-32800, "Request cancelled")
        self._diagnostic("Handler failed", exc)
        return RpcError(-32603, "Internal error")

    def _receive_response(self, message: dict[str, object]) -> None:
        request_id = message.get("id")
        # Do not let bool/float/string IDs alias an integer pending ID.
        if type(request_id) is not int:
            return
        with self._lock:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                return
            error = message.get("error")
            if message.get("jsonrpc") != "2.0" or (
                ("result" in message) == ("error" in message)
            ):
                pending.error = RpcError(-32603, "Invalid host response")
            elif "error" in message:
                if (
                    not isinstance(error, dict)
                    or type(error.get("code")) is not int
                    or not isinstance(error.get("message"), str)
                ):
                    pending.error = RpcError(-32603, "Invalid host response")
                else:
                    pending.error = RpcError(
                        error["code"], error["message"], error.get("data")
                    )
            else:
                pending.result = message["result"]
            pending.ready.set()

    def _reply(
        self,
        request_id: object,
        *,
        result: object = None,
        error: RpcError | None = None,
        allow_closing: bool = False,
    ) -> None:
        message: dict[str, object] = {"jsonrpc": "2.0", "id": request_id}
        if error is None:
            message["result"] = result
        else:
            detail: dict[str, object] = {"code": error.code, "message": error.message}
            if error.data is not None:
                detail["data"] = error.data
            message["error"] = detail
        try:
            try:
                self._send(message, allow_closing=allow_closing)
            except (TypeError, ValueError, OverflowError, RecursionError) as exc:
                self._diagnostic("Response is not JSON serializable", exc)
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": "Internal error"},
                    },
                    allow_closing=allow_closing,
                )
        except Cancelled:
            pass  # The transport is closed; in particular, suppress late worker replies.

    def _send(
        self,
        message: dict[str, object],
        *,
        allow_closing: bool = False,
        cancel_events: tuple[threading.Event, ...] = (),
    ) -> None:
        line = (
            json.dumps(
                message, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
            + "\n"
        )
        with self._write_lock, self._lock:
            if self._failed or (self._closing and not allow_closing):
                raise Cancelled("Transport closed")
            if any(event.is_set() for event in cancel_events):
                raise Cancelled("Request cancelled")
            try:
                written = self._writer.write(line)
                if written is not None and written != len(line):
                    raise OSError("Incomplete JSON-RPC write")
                self._writer.flush()
            except Exception as exc:
                self._failed = True
                self._closing = True
                self._cancel_waits_locked()
                self._diagnostic("Output failed", exc)
                raise Cancelled("Transport output failed") from exc

    def _cancel_waits_locked(self) -> None:
        self._run_cancel.set()
        for pending in self._pending.values():
            pending.error = Cancelled("Request cancelled")
            pending.ready.set()
        self._pending.clear()

    def _stop(self, handler: Handler) -> None:
        # Closing and output share the same lock order, so no worker line can
        # begin after this point. Only the explicit shutdown reply is allowed.
        with self._write_lock, self._lock:
            if self._cleanup_started:
                return
            self._cleanup_started = True
            self._closing = True
            self._cancel_waits_locked()
            worker = self._worker
        self._invoke(handler, "cancel", {})
        if worker is not None and worker is not threading.current_thread():
            worker.join(self._JOIN_TIMEOUT)
            if worker.is_alive():
                self._diagnostic("Run worker did not stop within the join timeout")

    def _is_closing(self) -> bool:
        with self._lock:
            return self._closing

    @staticmethod
    def _validate_outgoing(method: str, params: dict[str, object]) -> None:
        if not isinstance(method, str) or not isinstance(params, dict):
            raise ValueError("method must be a string and params must be an object")

    @staticmethod
    def _diagnostic(message: str, exc: Exception | None = None) -> None:
        try:
            print(f"[jsonrpc] {message}" + (f": {exc}" if exc else ""), file=sys.stderr)
        except Exception:
            pass
