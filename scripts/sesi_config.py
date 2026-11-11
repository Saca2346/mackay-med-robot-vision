"""
Session config: a copy of the camera config whose outputs (screens, event pictures, logs) go into the session folder,
so a test session never writes into another folder's evidence.

  python scripts/sesi_config.py config/box_pipeline_config_webcam.yaml results/sesi_X
  -> results/sesi_X/config_sesi.yaml
"""
import argparse
import pathlib
import sys

import yaml

def create_session_config(src: pathlib.Path, folder: pathlib.Path) -> pathlib.Path:
    """Keep pipeline settings, redirect every configured output to this session.

    PICK crops and grasp.jsonl are derived from the screens / verification paths
    by test_webcam_box.py, so these redirects also isolate those artifacts.
    """
    with pathlib.Path(src).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    output = cfg.setdefault("output", {})
    if not isinstance(output, dict):
        raise ValueError("config output must be a YAML mapping")
    folder = pathlib.Path(folder).resolve()
    output.update(screens_dir=str(folder / "screens"),
                  log_csv=str(folder / "selection_log.csv"),
                  verify_log_csv=str(folder / "verify_log.csv"))
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "config_sesi.yaml"
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("folder", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        target = create_session_config(args.source, args.folder)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"Session config failed: {exc}", file=sys.stderr)
        return 1
    print(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
