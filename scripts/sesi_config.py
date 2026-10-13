"""
Session config: a copy of the RealSense config whose outputs (screens, event pictures, logs) go into the session folder,
so a test session never writes into another folder's evidence.

  python scripts/sesi_config.py results/bukti_laporan_20261008/config_realsense.yaml results/sesi_X
  -> results/sesi_X/config_sesi.yaml
"""
import pathlib
import sys

import yaml

src, folder = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["output"]["screens_dir"] = str(folder / "screens")
cfg["output"]["log_csv"] = str(folder / "selection_log.csv")
cfg["output"]["verify_log_csv"] = str(folder / "verify_log.csv")
folder.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(cfg, open(folder / "config_sesi.yaml", "w", encoding="utf-8"), sort_keys=False)
print(folder / "config_sesi.yaml")
