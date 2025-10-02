from __future__ import annotations
import os, signal, subprocess, threading, time
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from app.config import Config

_lock = threading.RLock()
_proc: Optional[subprocess.Popen] = None
_watcher: Optional[threading.Thread] = None
_active_visit_id: Optional[int] = None
_on_finished: Optional[Callable[[Dict[str, Any]], None]] = None  # задаётся извне

def set_on_finished(cb: Callable[[Dict[str, Any]], None]) -> None:
    global _on_finished
    _on_finished = cb

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _abspath(p: str, root: Path) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (root / path).resolve())

def _is_executable(p: str) -> bool:
    return os.path.isfile(p) and os.access(p, os.X_OK)

def start_emulator(
    visit_id: int,
    *,
    binary_path: str = "motet/build/motet",
    bpm_csv: str = "motet/data/bpm_1.csv",
    uterus_csv: str = "motet/data/uterus_1.csv",
    url: Optional[str] = None,
    loop: bool = False,
    project_root: Optional[str] = None,
    force_restart: bool = False,
) -> int:
    """Стартует ОДИН экземпляр. Возвращает pid. Эмулятор постит в /api/monitor."""
    global _proc, _watcher, _active_visit_id
    with _lock:
        if _proc and _proc.poll() is None:
            if not force_restart:
                _active_visit_id = int(visit_id)
                return _proc.pid
            stop_emulator(graceful=True)

        _active_visit_id = int(visit_id)

        root = Path(project_root).resolve() if project_root else _project_root()
        binary_abs     = _abspath(binary_path, root)
        bpm_csv_abs    = _abspath(bpm_csv, root)
        uterus_csv_abs = _abspath(uterus_csv, root)

        if url is None:
            host = str(Config.FLASK_RUN_HOST).strip()
            port = str(Config.FLASK_RUN_PORT).strip()
            url = f"{host}:{port}/api/monitor"

        for fp in (bpm_csv_abs, uterus_csv_abs):
            if not Path(fp).exists():
                raise FileNotFoundError(f"Не найден файл данных: {fp}")

        if _is_executable(binary_abs):
            cmd = [binary_abs, "-bpm", bpm_csv_abs, "-uterus", uterus_csv_abs, "-url", url]
            if loop:
                cmd.append("-loop")
        else:
            # минимальный автопад: если бинаря нет — пробуем `go run motet/main.go`
            main_go_abs = _abspath("motet/main.go", root)
            if not Path(main_go_abs).exists():
                raise FileNotFoundError(f"Не найден бинарь {binary_abs} и motet/main.go")
            cmd = ["go", "run", main_go_abs, "-bpm", bpm_csv_abs, "-uterus", uterus_csv_abs, "-url", url]
            if loop:
                cmd.append("-loop")

        _proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            close_fds=True,
            start_new_session=(os.name != "nt"),
            stdout=None, stderr=None,
        )

        def _watch():
            rc = None
            try:
                if _proc:
                    rc = _proc.wait()
            finally:
                payload = {"visit_id": _active_visit_id, "returncode": rc, "finished_at": time.time()}
                if _on_finished:
                    try: _on_finished(payload)
                    except Exception: pass
                with _lock:
                    _cleanup_locked()

        _watcher = threading.Thread(target=_watch, daemon=True)
        _watcher.start()
        return _proc.pid

def stop_emulator(graceful: bool = True) -> bool:
    """Посылаем сигнал процессу. Финализация и очистка — в watcher."""
    with _lock:
        if not _proc or _proc.poll() is not None:
            return False
        try:
            if os.name == "nt":
                _proc.terminate()
            else:
                if graceful:
                    try: os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
                    except Exception: _proc.terminate()
                else:
                    try: os.killpg(os.getpgid(_proc.pid), signal.SIGKILL)
                    except Exception: _proc.kill()
            return True
        except Exception:
            return False

def emulator_status() -> Dict[str, Any]:
    with _lock:
        if not _proc:
            return {"running": False, "visit_id": _active_visit_id}
        rc = _proc.poll()
        return {"running": rc is None, "pid": _proc.pid, "returncode": rc, "visit_id": _active_visit_id}

def get_active_visit() -> Optional[int]:
    with _lock:
        return _active_visit_id

@contextmanager
def run_emulator(visit_id: int, **kwargs):
    pid = start_emulator(visit_id, **kwargs)
    try:
        yield pid
    finally:
        stop_emulator(graceful=True)

def _cleanup_locked() -> None:
    global _proc, _watcher, _active_visit_id
    _proc = None
    _watcher = None
    _active_visit_id = None
