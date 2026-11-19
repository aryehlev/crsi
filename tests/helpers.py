from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from toybench import SEED, FakePolicy, ToyBench, harness_with  # noqa: E402

from crsi import GreedySelector, Manifest, ProposalError, RunConfig, continue_rsi, run_source  # noqa: E402
from crsi.harness import apply_edits  # noqa: E402

ALL_A = ["reverse-0", "reverse-1", "strip-0", "strip-1", "upper-0", "upper-1"]
ROOT_CAUSE = "the harness returns the input unchanged for task kinds it has no handler for"


class ScriptedProposer:
    """Stands in for the optimizer model: a fixed plan of {(round, sample): step}.

    A step is {"manifest": {...}, "files": {...}}, an exception to raise, or absent (no proposal).
    """

    def __init__(self, plan: dict) -> None:
        self.plan = plan
        self.contexts: list = []

    def propose(self, ctx, sample, workdir):
        self.contexts.append(ctx)
        step = self.plan.get((ctx.round, sample))
        if step is None:
            raise ProposalError("nothing to propose")
        if isinstance(step, Exception):
            raise step
        apply_edits(workdir / "harness", step.get("files", {}), step.get("deletes"))
        return Manifest.from_dict(step["manifest"])


class ScriptedLLM:
    """A model that returns queued replies and remembers what it was asked."""

    def __init__(self, *replies: str, name: str = "scripted") -> None:
        self.replies = list(replies)
        self.name = name
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def step(what: str, kinds: tuple[str, ...], predicted: list[str], problem_id=None, root_cause: str = "", why: str = "") -> dict:
    return {
        "manifest": {"problem_id": problem_id, "root_cause": root_cause, "what": what, "why": why, "predicted": predicted},
        "files": {"harness.py": harness_with(*kinds)},
    }


def config(tmp_path: Path, name: str, **overrides) -> RunConfig:
    settings = dict(name=name, out=str(tmp_path / name), rounds=1, candidates=2, rollouts=1, verbose=False, timeout=None)
    settings.update(overrides)
    return RunConfig(**settings)



def make_source(tmp_path: Path, seed: Path, **overrides) -> dict:
    """Source RSI on toy-A: sample 0 proposes nothing, sample 1 adds a strip handler and
    predicts it will fix every task."""
    plan = {(1, 1): step("add a strip handler", ("strip",), ALL_A, root_cause=ROOT_CAUSE, why="strip tasks come back with their padding")}
    return run_source(ToyBench(FakePolicy()), seed, ScriptedProposer(plan), GreedySelector(), config(tmp_path, "src", **overrides))



# -- a local stand-in for the model APIs ----------------------------------------------------


class FakeModelServer:
    """Serves /v1/messages (Anthropic, streamed) and /chat/completions (OpenAI style).

    `reply(system, user) -> str` produces the text. `failures` is a queue of HTTP statuses, or
    callables writing a custom response, consumed one per request before any normal reply.
    """

    def __init__(self) -> None:
        self.reply = lambda system, user: "ok"
        self.failures: list = []
        self.requests: list[dict] = []
        self.stop_reason = "end_turn"
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "payload": payload})
                if outer.failures:
                    failure = outer.failures.pop(0)
                    if callable(failure):
                        return failure(self)
                    return self._json(failure, {"type": "error", "error": {"type": "overloaded_error" if failure == 529 else "invalid_request_error", "message": "scripted failure"}})
                if self.path.endswith("/v1/messages"):
                    system = payload.get("system", "")
                    text = outer.reply(system, payload["messages"][-1]["content"])
                    return self._sse(text, outer.stop_reason)
                messages = payload["messages"]
                system = next((m["content"] for m in messages if m["role"] == "system"), "")
                text = outer.reply(system, messages[-1]["content"])
                self._json(200, {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

            def _json(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _sse(self, text: str, stop_reason: str) -> None:
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                half = len(text) // 2
                events = [
                    {"type": "message_start", "message": {"id": "msg_1", "type": "message"}},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hidden"}},
                    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[:half]}},
                    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text[half:]}},
                    {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
                for event in events:
                    self.wfile.write(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

