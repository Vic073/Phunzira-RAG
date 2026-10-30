"""HTTP API. Run with ``phunzira serve`` (needs the ``api`` extra)."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .pipeline import StudyAssistant, build_assistant


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(4, ge=1, le=10)


class SourceOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    lang: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    question: str
    language: str
    answer: str
    sources: list[SourceOut]


def create_app(assistant: StudyAssistant | None = None) -> FastAPI:
    app = FastAPI(title="Phunzira", version=__version__)
    state: dict[str, StudyAssistant | None] = {"assistant": assistant}

    def get_assistant() -> StudyAssistant:
        if state["assistant"] is None:
            state["assistant"] = build_assistant()
        return state["assistant"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            result = get_assistant().ask(request.question, k=request.k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AskResponse(
            question=result.question,
            language=result.language,
            answer=result.text,
            sources=[SourceOut(**vars(s)) for s in result.sources],
        )

    return app
