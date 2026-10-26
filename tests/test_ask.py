import sys
from types import SimpleNamespace

from app import ask as ask_module


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
    )

    class FakeClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, **kwargs):
            return response

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("question") == "answer"
    assert capsys.readouterr().out.splitlines() == [
        "[入力トークン数] 12",
        "[出力トークン数] 7",
    ]


def test_ask_omits_generation_config_when_temperature_is_not_set(monkeypatch):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    call = {}

    class FakeClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, **kwargs):
            call.update(kwargs)
            return response

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("question") == "answer"
    assert call == {"model": ask_module.MODEL, "contents": "question"}


def test_ask_passes_explicit_temperature(monkeypatch):
    response = SimpleNamespace(text="answer", usage_metadata=None)
    call = {}

    class FakeClient:
        def __init__(self, api_key):
            self.models = self

        def generate_content(self, **kwargs):
            call.update(kwargs)
            return response

    monkeypatch.setattr(ask_module.genai, "Client", FakeClient)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")

    assert ask_module.ask("question", temperature=0.4) == "answer"
    assert call == {
        "model": ask_module.MODEL,
        "contents": "question",
        "config": {"temperature": 0.4},
    }


def test_main_accepts_temperature_option(monkeypatch, capsys):
    call = {}
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.ask", "LoRA とは何ですか？", "--temperature", "0.4"],
    )
    monkeypatch.setattr(
        ask_module,
        "ask",
        lambda question, temperature=None: (
            call.update(question=question, temperature=temperature) or "answer"
        ),
    )

    ask_module.main()

    assert call == {"question": "LoRA とは何ですか？", "temperature": 0.4}
    output = capsys.readouterr().out
    assert "[temperature] 0.4" in output
    assert "answer" in output


def test_main_displays_default_temperature_when_not_specified(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["app.ask", "question"])
    monkeypatch.setattr(ask_module, "ask", lambda question, temperature=None: "answer")

    ask_module.main()

    assert "[temperature] 省略（モデル/APIのデフォルト）" in capsys.readouterr().out
