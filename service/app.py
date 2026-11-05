"""FastAPI case note service. API described in docs/api.md.

Run:  .\\.venv\\Scripts\\python.exe -m uvicorn service.app:app --port 8000
"""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from service.llm import generator_from_env
from service.notes import NoteService
from service.schemas import Case, NoteResult


def create_app(notes: NoteService) -> FastAPI:
    app = FastAPI(title="Trade Surveillance case notes", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm": "on" if notes.llm is not None else "off"}

    @app.post("/v1/notes", response_model=NoteResult)
    def make_note(case: Case) -> NoteResult:
        """Write a case note for one alert. Always returns a note (LLM or template)."""
        return notes.make_note(case)

    return app


def _default_app() -> FastAPI:
    load_dotenv()  # reads the local .env (LLM_PROVIDER, LLM_MODEL, OLLAMA_URL) if present
    return create_app(NoteService(generator_from_env()))


app = _default_app()
