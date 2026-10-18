from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .loader import UnsupportedFormat
from .pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline(get_settings())
    yield


app = FastAPI(title="RAG: upload → chunk → embed → ChromaDB → retrieve → answer",
              lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def _pipeline() -> RAGPipeline:
    return app.state.pipeline


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    where: dict | None = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=50)
    where: dict | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> dict:
    return _pipeline().stats()


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix
    if not suffix:
        raise HTTPException(415, "file must have an extension (.pdf/.html/.md)")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        return _pipeline().ingest_file(tmp_path)
    except UnsupportedFormat as e:
        raise HTTPException(415, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest) -> dict:
    try:
        return _pipeline().retrieve(req.query, top_k=req.top_k, where=req.where)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return _pipeline().ask(req.question, top_k=req.top_k, where=req.where)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))