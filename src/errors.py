"""Errors shared by the stdio transport and plugin application."""


class RpcError(Exception):
    """A JSON-RPC error safe to send to the caller."""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class Cancelled(Exception):
    """Cooperative cancellation of the active run or host-service wait."""
