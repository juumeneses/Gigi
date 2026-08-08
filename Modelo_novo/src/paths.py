from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent


def repo_path(value: str | Path | None) -> Path | None:
    """Resolve caminhos relativos e recupera caminhos absolutos de outro clone."""
    if value is None or str(value).strip() in {"", "nan", "None"}:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        return (REPO_ROOT / path).resolve()
    if path.exists():
        return path
    parts = path.parts
    for marker in ("Treinamento_faces", "Treinamento_cranios", "Matching_cranio_face", "Modelo_novo"):
        if marker in parts:
            return REPO_ROOT.joinpath(*parts[parts.index(marker):])
    return path


def relative_repo_path(value: str | Path | None) -> str:
    path = repo_path(value)
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def ensure_project_dirs() -> None:
    for rel in ("data/manifests", "data/splits", "data/processed", "checkpoints", "reports", "resultados", "runs"):
        (PROJECT_DIR / rel).mkdir(parents=True, exist_ok=True)

