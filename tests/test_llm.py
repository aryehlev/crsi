import pytest

from crsi import LLMError, LLMRefusal, make_llm


def client(spec="anthropic:claude-haiku-5-5", **options):
    return make_llm(spec, backoff=0.0, **options)


def test_anthropic_request_shape_and_streamed_reply(model_server):
    model_server.reply = lambda system, user: f"{system}|{user}|done"
    llm = client(max_tokens=1234, effort="low")
    assert llm.name == "anthropic:claude-haiku-5-5"
    assert llm.complete("be brief", "hello") == "be brief|hello|done"  # text deltas only, thinking is skipped
    request = model_server.requests[0]
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "test-key" and request["headers"]["anthropic-version"] == "2026-10-21"
    assert request["payload"] == {
        "model": "claude-haiku-5-5", "max_tokens": 1234, "stream": True, "system": "be brief",
        "messages": [{"role": "user", "content": "hello"}], "output_config": {"effort": "low"},
    }
    client().complete("", "no system")
    later = model_server.requests[1]["payload"]
    assert "system" not in later and "temperature" not in later and "thinking" not in later and "output_config" not in later


def test_overload_is_retried_and_a_bad_request_is_not(model_server):
    model_server.failures = [529, 500]
    assert client().complete("s", "u") == "ok" and len(model_server.requests) == 3
    model_server.failures = [400, 200]
    with pytest.raises(LLMError, match="HTTP 400: invalid_request_error: scripted failure"):
        client().complete("s", "u")
    assert len(model_server.requests) == 4  # no retry after the 400
    model_server.failures = [529] * 3
    with pytest.raises(LLMError, match="gave up after 3 attempt"):
        client(retries=2).complete("s", "u")


def test_errors_inside_the_stream_and_broken_streams_are_retried(model_server):
    def overloaded(handler):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}\n\n')

    def cut_off(handler):
        handler.send_response(200)
        handler.send_header("content-type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(b'event: content_block_delta\ndata: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "par"}}\n\n')

    model_server.failures = [overloaded, cut_off]
    assert client().complete("s", "u") == "ok" and len(model_server.requests) == 3


def test_a_refusal_or_an_empty_reply_is_not_a_connection_error(model_server):
    model_server.stop_reason = "refusal"
    with pytest.raises(LLMRefusal, match="declined"):
        client().complete("s", "u")
    model_server.stop_reason = "end_turn"
    model_server.reply = lambda system, user: "  "
    with pytest.raises(LLMRefusal, match="no text"):
        client().complete("s", "u")
    assert len(model_server.requests) == 2  # neither is retried
    model_server.stop_reason = "max_tokens"
    model_server.reply = lambda system, user: "cut off mid"
    assert client().complete("s", "u") == "cut off mid"


def test_openai_compatible_servers(model_server, monkeypatch):
    model_server.reply = lambda system, user: f"{system}/{user}"
    llm = client("openai:local-model")
    assert llm.name == "openai:local-model" and llm.complete("sys", "question") == "sys/question"
    request = model_server.requests[0]
    assert request["path"] == "/v1/chat/completions" and request["headers"]["authorization"] == "Bearer test-key"
    assert request["payload"] == {"model": "local-model", "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}]}
    monkeypatch.delenv("OPENAI_API_KEY")  # a local server needs no key
    assert client("openai:local-model", max_tokens=50).complete("", "q") == "/q"
    assert "authorization" not in model_server.requests[1]["headers"] and model_server.requests[1]["payload"]["max_tokens"] == 50
    model_server.failures = [429]
    assert client("openai:local-model").complete("", "q") == "/q"


def test_make_llm_checks_the_spec_and_the_credentials(monkeypatch):
    for bad in ("claude-haiku-5-5", "gemini:flash", "anthropic:"):
        with pytest.raises(ValueError, match="provider:model"):
            make_llm(bad)
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_llm("anthropic:claude-haiku-5-5")
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_llm("openai:gpt")
    existing = object.__new__(type("Custom", (), {"complete": lambda self, s, u: "x", "name": "custom"}))
    assert make_llm(existing) is existing
