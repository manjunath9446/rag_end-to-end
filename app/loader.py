import re
from dataclasses import dataclass
from pathlib import Path


class UnsupportedFormat(Exception):
    pass


@dataclass
class LoadedDoc:
    text: str
    filename: str
    extension: str


def _clean(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_md(path: Path) -> LoadedDoc:
    return LoadedDoc(_clean(_read(path)), path.name, "md")


def _load_html(path: Path) -> LoadedDoc:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(_read(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return LoadedDoc(_clean(soup.get_text(separator="\n")), path.name, "html")


def _load_pdf(path: Path) -> LoadedDoc:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
            
    joined = "\n\n".join(pages).strip()
    if not joined:
        raise ValueError(f"No text could be extracted from {path.name}. (Scanned PDFs are not supported).")

    return LoadedDoc(_clean(joined), path.name, "pdf")


_LOADERS = {
    ".md": _load_md,
    ".markdown": _load_md,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
}


def load(path: Path) -> LoadedDoc:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
        
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    
    if loader is None:
        raise UnsupportedFormat(
            f"{ext!r} is not supported. Allowed: {sorted(_LOADERS)}"
        )
        
    doc = loader(path)
    return doc