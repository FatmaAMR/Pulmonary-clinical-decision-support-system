import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any

ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")


def load_dotenv_file(env_path: Path | None = None) -> None:
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                val = v.strip().strip('"').strip("'")
                if val:
                    os.environ[k.strip()] = val


def _cast_scalar(val: str) -> Any:
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def _interpolate(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(x) for x in obj]
    if isinstance(obj, str):
        full_match = ENV_PATTERN.fullmatch(obj.strip())
        if full_match:
            var_name, default_val = full_match.group(1), full_match.group(2)
            raw = os.getenv(var_name, default_val if default_val is not None else "")
            return _cast_scalar(raw)

        def repl(m: re.Match) -> str:
            var_name, default_val = m.group(1), m.group(2)
            return os.getenv(var_name, default_val if default_val is not None else "")

        return ENV_PATTERN.sub(repl, obj)
    return obj


def load_pipeline_config(config_path: str | Path) -> Dict[str, Any]:
    load_dotenv_file()
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return _interpolate(raw_cfg)