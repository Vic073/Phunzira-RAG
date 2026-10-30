from types import SimpleNamespace

import pytest

from phunzira.chunking import Chunk
from phunzira.config import Config
from phunzira.generate import ClaudeGenerator, ExtractiveGenerator, make_generator
from phunzira.pipeline import StudyAssistant, build_assistant
from phunzira.retrieval import Hit


def _hits():
    chunk = Chunk(
        "soil-en#0",
        "soil-en",
        "Soil",
        "en",
        "Soil is the top layer of the ground. Farmers add manure to restore nutrients. Trees protect soil from erosion.",
    )
    other = Chunk("lake-en#0", "lake-en", "Lake", "en", "The lake is long.")
    return [Hit(chunk, 2.0), Hit(other, 1.0)]


def test_extractive_picks_overlapping_sentences():
    gen = ExtractiveGenerator(max_sentences=1).generate("What do farmers add?", _hits(), "en")
    assert "manure" in gen.text
    assert gen.cited == [1]


def test_extractive_handles_no_hits():
    gen = ExtractiveGenerator().generate("anything", [], "ny")
    assert gen.cited == []
    assert gen.text.startswith("Sindinapeze")


class _FakeClient:
    def __init__(self, reply):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)
        self._reply = reply

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply)])


def test_claude_generator_parses_and_filters_citations():
    client = _FakeClient("Farmers add manure [1]. Trees help too [2][7].")
    gen = ClaudeGenerator(model="test-model", client=client).generate("q", _hits(), "en")
    assert gen.cited == [1, 2]  # [7] does not exist and is dropped
    call = client.calls[0]
    assert call["model"] == "test-model"
    assert "English" in call["system"]
    assert "[1] (Soil, en)" in call["messages"][0]["content"]


def test_claude_generator_asks_for_chichewa_when_question_is_chichewa():
    client = _FakeClient("Yankho [1]")
    ClaudeGenerator(client=client).generate("q", _hits(), "ny")
    assert "Chichewa" in client.calls[0]["system"]


def test_make_generator_auto_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_generator("auto").name == "extractive"
    with pytest.raises(ValueError):
        make_generator("nonsense")


@pytest.fixture(scope="module")
def assistant() -> StudyAssistant:
    return build_assistant(Config(generator="extractive"))


def test_pipeline_end_to_end_english(assistant):
    answer = assistant.ask("How do plants make their own food?")
    assert answer.language == "en"
    assert answer.sources and answer.sources[0].doc_id == "photosynthesis-en"
    assert "sunlight" in answer.text.lower()


def test_pipeline_end_to_end_chichewa(assistant):
    answer = assistant.ask("Kodi zomera zimapanga bwanji chakudya chawo?")
    assert answer.language == "ny"
    assert answer.sources[0].doc_id.startswith("photosynthesis")


def test_pipeline_rejects_blank_question(assistant):
    with pytest.raises(ValueError):
        assistant.ask("   ")


def test_api_endpoints(assistant):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from phunzira.api import create_app

    client = TestClient(create_app(assistant))
    assert client.get("/health").json()["status"] == "ok"

    response = client.post("/ask", json={"question": "How do clouds form?", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "en"
    assert body["sources"][0]["doc_id"] == "water-cycle-en"

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={"question": "x", "k": 99}).status_code == 422
