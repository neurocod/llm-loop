"""Latest main-request context occupancy, never cumulative billed tokens."""
from __future__ import annotations


INPUT_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
DEFAULT_WARNING_PERCENT = 45.0


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _base(model: str) -> str:
    # A routing suffix identifies a configuration, not a measured window size.
    return model.split("[", 1)[0]


class ContextTracker:
    """Use Claude's statusline formula: input + cache creation + cache reads.

    https://code.claude.com/docs/en/statusline#context-window-fields specifies
    input-only usage from the latest API call; output and result.usage totals
    are excluded. modelUsage supplies capacity, matched to the primary model,
    not the auxiliary title-generation model that can appear first. An unknown
    capacity remains unknown even when the model name contains '[1m]'.
    """

    def __init__(self, prior: dict | None = None, warning_percent: float = DEFAULT_WARNING_PERCENT):
        prior = prior or {}
        self.model = prior.get("model") or ""
        self.input_tokens = _count(prior.get("input_tokens"))
        self.context_window = _count(prior.get("context_window")) or None
        self.warning_percent = warning_percent
        self._capacities: dict = {}

    def _set_model(self, model: str, *, from_message: bool = False) -> None:
        if not model or model == self.model:
            return
        if from_message and self.model and model == _base(self.model):
            return  # Keep the init event's exact routing configuration.
        self.model = model
        self.input_tokens = None
        self.context_window = None
        self._resolve_window()

    def _resolve_window(self) -> None:
        exact = self._capacities.get(self.model)
        if isinstance(exact, dict):
            self.context_window = _count(exact.get("contextWindow")) or None
            return
        matches = [row for row in self._capacities.values()
                   if isinstance(row, dict) and row.get("canonicalModel") == _base(self.model)]
        windows = {_count(row.get("contextWindow")) for row in matches}
        if len(windows) == 1 and next(iter(windows)):
            self.context_window = windows.pop()
        elif matches:
            self.context_window = None  # Ambiguous configurations are not measurements.

    def observe(self, event: dict) -> dict | None:
        """Return a changed snapshot, or None for irrelevant/duplicate events."""
        if not isinstance(event, dict) or event.get("parent_tool_use_id"):
            return None
        before = self.snapshot()
        kind = event.get("type")
        if kind == "system":
            if event.get("subtype") == "init":
                self._set_model(event.get("model") or "")
            elif event.get("subtype") == "compact_boundary" or (
                    event.get("subtype") == "status" and event.get("status") == "compacting"):
                self.input_tokens = None
        message = event.get("message") if kind == "assistant" else None
        if kind == "stream_event":
            inner = event.get("event") or {}
            if inner.get("type") == "message_start":
                message = inner.get("message")
        if isinstance(message, dict):
            self._set_model(message.get("model") or "", from_message=True)
            usage = message.get("usage")
            if isinstance(usage, dict) and "input_tokens" in usage:
                counts = [_count(usage.get(key, 0)) for key in INPUT_FIELDS]
                self.input_tokens = sum(counts) if None not in counts else None
        if kind == "result" and isinstance(event.get("modelUsage"), dict):
            self._capacities = event["modelUsage"]
            self._resolve_window()
        after = self.snapshot()
        return after if after != before else None

    def snapshot(self) -> dict:
        known = self.input_tokens is not None and self.context_window is not None
        percent = 100 * self.input_tokens / self.context_window if known else None
        return {"model": self.model or None, "input_tokens": self.input_tokens,
                "context_window": self.context_window, "used_percent": percent,
                "warning": percent is not None and percent >= self.warning_percent}
