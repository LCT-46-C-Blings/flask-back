from __future__ import annotations
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from app.config import Config

_lock = threading.RLock()
_proc: Optional[subprocess.Popen] = None
_active_visit_id: Optional[int] = None

def _project_root() -> Path:
    # .../fetality/app -> .../fetality
    return Path(__file__).resolve().parents[1]

def _abspath(p: str, root: Path) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else (root / path).resolve())

def _is_executable(p: str) -> bool:
    return os.path.isfile(p) and os.access(p, os.X_OK)

def start_emulator(
    visit_id: int,
    *,
    # --- основное: путь к уже собранному бинарю ---
    binary_path: str = "motet/build/motet",  # дефолт: относит. к корню проекта
    # --- данные и адрес приёмника ---
    bpm_csv: str = "motet/data/bpm_1.csv",
    uterus_csv: str = "motet/data/uterus_1.csv",
    url: str = Config.FLASK_RUN_HOST + ":" + Config.FLASK_RUN_PORT + "/api/monitor",
    loop: bool = False,
    # --- опциональный fallback на go run (если бинаря нет) ---
    allow_go_run_fallback: bool = True,
    main_go: str = "motet/main.go",
    # --- рабочая директория (если относительные пути сверху) ---
    project_root: Optional[str] = None,
    force_restart: bool = False
) -> int:
    """
    Стартует собранный эмулятор как дочерний процесс и привязывает поток к visit_id.
    Возвращает pid. Бросает RuntimeError, если уже запущен.

    Дефолты:
      binary_path="emulator/bin/emulator"
      bpm_csv="data/bpm.csv"
      uterus_csv="data/uterus.csv"
      url="localhost:5000"
      loop=False
      allow_go_run_fallback=True  (если бинаря нет — попробуем `go run main.go`)
      main_go="emulator/main.go"
    """
    global _proc, _active_visit_id
    with _lock:
        # если уже запущен
        if _proc and _proc.poll() is None:
            if force_restart:
                stop_emulator(graceful=True)
            else:
                # просто обновим visit и вернём текущий pid (идемпотентность)
                _active_visit_id = int(visit_id)
                return _proc.pid

        _active_visit_id = int(visit_id)

        root = Path(project_root).resolve() if project_root else _project_root()
        binary_abs     = _abspath(binary_path, root)
        bpm_csv_abs    = _abspath(bpm_csv, root)
        uterus_csv_abs = _abspath(uterus_csv, root)
        main_go_abs    = _abspath(main_go, root)

        # проверяем входные файлы
        for fp in (bpm_csv_abs, uterus_csv_abs):
            if not Path(fp).exists():
                raise FileNotFoundError(f"Не найден файл данных: {fp}")

        # готовим команду
        if _is_executable(binary_abs):
            cmd = [binary_abs, "-bpm", bpm_csv_abs, "-uterus", uterus_csv_abs, "-url", url]
            if loop:
                cmd.append("-loop")
        else:
            if not allow_go_run_fallback:
                raise FileNotFoundError(
                    f"Не найден исполняемый файл эмулятора: {binary_abs}. "
                    f"Либо соберите его, либо укажите binary_path, либо включите fallback на go run."
                )
            if not Path(main_go_abs).exists():
                raise FileNotFoundError(
                    f"Ни бинарь ({binary_abs}) ни main.go ({main_go_abs}) не найдены."
                )
            # fallback: go run
            cmd = ["go", "run", main_go_abs, "-bpm", bpm_csv_abs, "-uterus", uterus_csv_abs, "-url", url]
            if loop:
                cmd.append("-loop")

        # не перенаправляем stdout/err (совместимо с eventlet/gevent), лог уедет в stdout процесса
        _proc = subprocess.Popen(cmd, cwd=str(root), close_fds=True)
        return _proc.pid

def stop_emulator(graceful: bool = True) -> bool:
    """Останавливает эмулятор и сбрасывает active visit. Возвращает True, если процесс был живой."""
    global _proc, _active_visit_id
    with _lock:
        _active_visit_id = None
        if not _proc:
            return False
        if _proc.poll() is not None:
            _proc = None
            return False
        try:
            if os.name == "nt" or not graceful:
                _proc.terminate()
            else:
                _proc.send_signal(signal.SIGINT)
        finally:
            _proc = None
        return True

def emulator_status() -> Dict[str, Any]:
    with _lock:
        if _proc is None:
            return {"running": False, "visit_id": _active_visit_id}
        rc = _proc.poll()
        return {"running": rc is None, "pid": _proc.pid, "returncode": rc, "visit_id": _active_visit_id}

def get_active_visit() -> Optional[int]:
    with _lock:
        return _active_visit_id

@contextmanager
def run_emulator(visit_id: int, **kwargs):
    pid = start_emulator(visit_id, **kwargs)
    print(pid)
    try:
        yield pid
    finally:
        stop_emulator()
