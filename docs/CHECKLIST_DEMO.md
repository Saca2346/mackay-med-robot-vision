# Text-request demo checklist

Schedule: progress report **Friday 09-10 afternoon**, follow-up meeting + demo **Monday 12-10**.
Basis: meeting of 05-10. What counts is the running program. Input brand + product + spec; output product, spec and position
(rack / level). The final confirmation stays with the doctor.

Last updated: 08-10 (results from the hucenrotia-ai server + Codex audit + K10 back on the shelf). This is the status of
that day; newer work is in `docs/FIXES_LOG_20261008-11.md` and `docs/HANDOFF.md`.

## Done

- [x] Meeting summary recorded. Claims that appear only in the AI summary are marked as not meeting decisions:
      dual arm, RealSense, expiry-date OCR, barcode as a fallback. The RealSense D455 is in use now (since 08-10), but that is a
      hardware choice, not a meeting decision.
- [x] `src/box_demo.py`: position table, `lookup()`, five statuses, TM message (`to_tm_message`), screen text and JSONL log.
- [x] `data/rak/posisi.csv`: **EXAMPLE**. Rack R1, level 1, column = order on the shelf. Not from the team yet.
- [x] `scripts/demo_request.py`: interactive mode, `--sekali` and `--json`. The inventory is only read, never changed.
- [x] `scripts/buat_paket_server.py`: fixed file list, sha256 per file, refuses to build a package when a `src` module is missing.
- [x] `pick_box()` uses **strict FEFO** (Wardana's decision, option A): always the box that expires first.
      If its position is not in the table, the status becomes NEEDS CONFIRMATION. The system does not switch to another box.
- [x] `tests/test_box_demo.py`: 11 tests. All tests on the laptop: **205 passed** (268 as of 14-10).
- [x] Demo scenarios tested on the laptop: all six as expected. One request per status:

  | Request | Expected result |
  | --- | --- |
  | `ivascular angiolite 2.5 x 29` | READY, K15 |
  | `angiolite 4x19` | READY, K10 (since 08-10 K10 is back on the shelf; before, NOT FOUND while K10 was taken) |
  | `angiolite 3 x 20` | NOT FOUND (not in the inventory) |
  | `angiolite 2.5` | REPEAT (length missing) |
  | `sled bag` | READY, the box with the earliest expiry (item without a size) |
  | `terumo xperience 2.5 x 15` | REPEAT (wrong brand) |

- [x] Server package built: `box_pipeline_demo_20261007.zip`.

## Not yet (status 08-10)

- [x] Run `python scripts/uji_checklist.py` on the server: **20 passed, 0 failed** (08-10, before K10 came back;
      `results/uji_checklist_20261007_235241.md`). After K10 came back (08-10, afternoon): **19 passed, 0 failed, 1 skipped**
      (#17 skipped automatically while K10 is on the shelf, see tables B / A).
- [x] Code via `git clone`, data via `data_demo_20261007.zip` (36/36 files match), model `models/box_v4s.pt`
      (sha256 `21c4bf01...`). `pytest` on the server: **205 passed**. Needs `pip install pytesseract==0.3.*` in `trse_venv`.
- [x] Text rehearsal on the server: the six demo scenarios give the same results as on the laptop.
- [ ] Demo rehearsal through Remmina, recording the screen as a fallback.
- [x] Camera mode in the demo program: `--kamera` hands a READY status to `test_webcam_box.py` (tested up to the camera script
      opening; not yet with a real camera / stream).
- [ ] Real camera-mode test with the RealSense D455 (see `docs/LIVE_TEST_LOCAL.md` for laptop only). Attempt of 08-10 on the
      laptop with 20 boxes in one frame: 0 PICK in 145 seconds (text too small); the camera needs to be closer, not retested yet.
- [ ] Friday report: draft in `docs/archive_id/LAPORAN_PROGRES_20261009.md` (English: `docs/PROGRESS_REPORT_20261009.md`).
- [x] v6 trained and evaluated (08-10). **Important correction:** the CV score that is really comparable with v4 is mAP50-95
      **0.706 +/- 0.032** (v4: 0.696 +/- 0.044): a small gain, not a big jump. Real box detection (out-of-fold): v4 100 % (123/123),
      v6 95.1 % (117/123, 4 photos even used the fallback weights). 0 wrong MATCH still holds. **Not used**; the demo stays on v4.
- [ ] Replace the example `posisi.csv` with rack / level data from the team.
- [ ] The Codex audit (4 Oct, `prelabel_size_label_full`) found 8 issues in the label pipeline: 7 photos still "to be checked by
      hand" but counted for training, 25 consecutive photo pairs leaking between train / val, and a training cache that is not
      always invalidated when an experiment changes. Details: `B_prelabel_perbaikan_tools/LAPORAN_AUDIT.md` (package 08-10).

## The 20 tests (fill in Result and OK while testing)

Automatic (text tests 1-20, report to `results/`): `python scripts/uji_checklist.py`. Results on laptop and server
(before K10 came back): **20 passed, 0 failed**. Since K10 came back (08-10): **19 passed, 0 failed, 1 skipped** (#17).
Camera per number: `python scripts/uji_checklist.py --kamera N` (N = 1-16, physical box in front of the camera; N = 17
also works while K10 is on the shelf, see `TEST_LIVE_LOCAL.bat` / `docs/LIVE_TEST_LOCAL.md`).

A. Item on the shelf: must be **READY** (text), then a camera test for numbers 1-17.

| # | Request | Box | Note | Text result | Camera result | OK |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `angiolite 2.5 x 29` | K15 | since 08-10 K10 (angiolite 4 x 19) is back on the shelf: look-alike pair, see #17 | READY K15 (server, PASS) |  | [ ] |
| 2 | `xperience pro 2.5 x 15` | K06 |  | READY K06 (server, PASS) |  | [ ] |
| 3 | `essential pro 3 x 40` | K04 | length size | READY K04 (server, PASS) |  | [ ] |
| 4 | `accuforce 2.5 x 12` | K07 | look-alike of #5/#6 | READY K07 (server, PASS) |  | [ ] |
| 5 | `accuforce 2.75 x 20` | K16 | look-alike of #4/#6 | READY K16 (server, PASS) |  | [ ] |
| 6 | `accuforce 3.5 x 15` | K21 | look-alike of #4/#5 | READY K21 (server, PASS) |  | [ ] |
| 7 | `ryurei 2 x 10` | K09 | smallest size | READY K09 (server, PASS) |  | [ ] |
| 8 | `nagomi 2.75 x 33` | K13 | look-alike of #9 | READY K13 (server, PASS) |  | [ ] |
| 9 | `nagomi 3 x 24` | K18 | expiry 2028-01-31 | READY K18 (server, PASS) |  | [ ] |
| 10 | `nc emerge 5.5 x 12` | K08 | expiry 2028-01-03 | READY K08 (server, PASS) |  | [ ] |
| 11 | `conqueror 3.5 x 15` | K11 | look-alike of #12 | READY K11 (server, PASS) |  | [ ] |
| 12 | `conqueror 4 x 8` | K12 | look-alike of #11 | READY K12 (server, PASS) |  | [ ] |
| 13 | `sapphire 3 x 15` | K14 |  | READY K14 (server, PASS) |  | [ ] |
| 14 | `scoreflex 3 x 15` | K19 |  | READY K19 (server, PASS) |  | [ ] |
| 15 | `agent 2.75 x 30` | K20 |  | READY K20 (server, PASS) |  | [ ] |
| 16 | `permanent sled bag` | K03 | 4 boxes (K01, K03, K05, K17); FEFO test: 2029-03-10 expires first | READY K03 (server, PASS) |  | [ ] |
| 17 | `angiolite 4 x 19` | K10 | since 08-10 K10 is back on the shelf; look-alike of #1, the riskiest mix-up | READY K10 (laptop, PASS, 08-10) |  | [ ] |

B. Must be refused or not found (text test only, no camera).

| # | Request | Expected result | Note | Text result | OK |
| --- | --- | --- | --- | --- | --- |
| 18 | `accuforce 2.75 x 15` | NOT FOUND, combination does not exist | look-alike of #5 and #6 | NOT FOUND (server, PASS) | [x] |
| 19 | `terumo xperience 2.5 x 15` | REPEAT, brand does not match the product |  | REPEAT (server, PASS) | [x] |
| 20 | `essential pro 3` | REPEAT, length missing |  | REPEAT (server, PASS) | [x] |

Camera pass rule: never a MATCH on a wrong box. When in doubt the result must be NEEDS CONFIRMATION.

## Questions for the team

1. Format of the rack / level position file: which columns, and per box or per product?
2. What data form does the TM software accept: JSON, text, or socket / ROS? For now we use `to_tm_message()`
   (example in `docs/PROGRESS_REPORT_20261009.md` section 1): a **temporary** format, not confirmed by the team.
3. ~~Who owns the voice confirmation interface~~ **Decision 08-10: use the TEXT interface first for the demo**
   (`demo_request.py`); the voice UI (colour + "execute") waits until the team assigns an owner.

## Commands

Laptop, cmd, in the code folder:

```bat
venv\Scripts\python.exe -m pytest tests/test_box_demo.py -q
venv\Scripts\python.exe scripts\demo_request.py
venv\Scripts\python.exe scripts\buat_paket_server.py
scp C:\Users\sacab\Downloads\box_pipeline_demo_20261007.zip Zen@140.113.149.94:~/Downloads/
```

Server, terminal (now preferably through VS Code Remote-SSH, see `docs/REMOTE_SERVER_CLAUDE_CODEX.md`):

```bash
cd ~/Downloads/box_pipeline && unzip -o ~/Downloads/box_pipeline_demo_20261007.zip
sha256sum -c --quiet paket_sha256.txt
source ~/trse_venv/bin/activate
python -m pytest tests -q
python scripts/demo_request.py
```

The date in the zip name follows the day the package was built. Use the name printed by `buat_paket_server.py`.
