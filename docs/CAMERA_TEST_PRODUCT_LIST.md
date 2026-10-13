# Product list for camera tests / recording rehearsal

Hold the matching physical box in front of the camera, say / type the **request**, and compare the result with the
**expected box**. The status must be **READY**. Run them one at a time with:

```bash
python scripts/demo_request.py --sekali "REQUEST" --kamera
# or by checklist number:  python scripts/uji_checklist.py --kamera N   (N = 1-16)
```

Pass rule: never a MATCH on a wrong box. Unsure = NEEDS CONFIRMATION (that is normal, not a failure).
In the demo menu (`DEMO.bat`, menu 3 / L) the same boxes have their own list numbers, one per box; see `docs/TEST_SESSION_GUIDE.md`.

## A. Item on the shelf (camera test, must be READY)

| # | Request (say / type) | Brand | Product | Size (diameter x length, mm) | Expiry | Box | Position |
|---|---|---|---|---|---|---|---|
| 1 | `ivascular angiolite 2.5 x 29` | iVascular | Angiolite | 2.5 x 29 | - | K15 | R1 / level 1 / column 15 |
| 2 | `ivascular xperience pro 2.5 x 15` | iVascular | Xperience Pro | 2.5 x 15 | - | K06 | R1 / level 1 / column 6 |
| 3 | `ivascular essential pro 3 x 40` | iVascular | Essential Pro | 3 x 40 | - | K04 | R1 / level 1 / column 4 |
| 4 | `terumo accuforce 2.5 x 12` | Terumo | Accuforce | 2.5 x 12 | - | K07 | R1 / level 1 / column 7 |
| 5 | `terumo accuforce 2.75 x 20` | Terumo | Accuforce | 2.75 x 20 | - | K16 | R1 / level 1 / column 16 |
| 6 | `terumo accuforce 3.5 x 15` | Terumo | Accuforce | 3.5 x 15 | - | K21 | R1 / level 1 / column 21 |
| 7 | `terumo ryurei 2 x 10` | Terumo | Ryurei | 2 x 10 | - | K09 | R1 / level 1 / column 9 |
| 8 | `terumo ultimaster nagomi 2.75 x 33` | Terumo | Ultimaster Nagomi | 2.75 x 33 | - | K13 | R1 / level 1 / column 13 |
| 9 | `terumo ultimaster nagomi 3 x 24` | Terumo | Ultimaster Nagomi | 3 x 24 | 2028-01-31 | K18 | R1 / level 1 / column 18 |
| 10 | `boston scientific nc emerge 5.5 x 12` | Boston Scientific | NC Emerge | 5.5 x 12 | 2028-01-03 | K08 | R1 / level 1 / column 8 |
| 11 | `apt medical conqueror nc pro 3.5 x 15` | APT Medical | Conqueror NC Pro | 3.5 x 15 | - | K11 | R1 / level 1 / column 11 |
| 12 | `apt medical conqueror nc pro 4 x 8` | APT Medical | Conqueror NC Pro | 4 x 8 | - | K12 | R1 / level 1 / column 12 |
| 13 | `orbusneich sapphire 3 x 15` | OrbusNeich | Sapphire | 3 x 15 | - | K14 | R1 / level 1 / column 14 |
| 14 | `orbusneich scoreflex 3 x 15` | OrbusNeich | Scoreflex | 3 x 15 | - | K19 | R1 / level 1 / column 19 |
| 15 | `boston scientific agent 2.75 x 30` | Boston Scientific | Agent | 2.75 x 30 | - | K20 | R1 / level 1 / column 20 |
| 16 | `boston scientific permanent sled bag` | Boston Scientific | Permanent Sled Bag | no size | 2029-03-10 | K03 | R1 / level 1 / column 3 |

Note on test #16: the inventory has 4 Permanent Sled Bag boxes (K01, K03, K05, K17) with different expiry dates. The
system must choose **K03** because it expires first (2029-03-10): this tests the FEFO rule, not just "the first box found".
Since 14 Oct a date can also be requested for one specific box, e.g. `permanent sled bag 2029-03-24` (K05); that box is
picked only after its printed date has been read.

## B. Must be refused or not found (text only, no camera)

| # | Request | Expected result | Why |
|---|---|---|---|
| 17 | `ivascular angiolite 4 x 19` | NOT FOUND | already taken (K10): valid only while K10 is marked `diambil` (taken) |
| 18 | `terumo accuforce 2.75 x 15` | NOT FOUND | close to #5 and #6, this combination is not in the inventory |
| 19 | `terumo xperience 2.5 x 15` | REPEAT | Xperience Pro is not a Terumo product (it is iVascular) |
| 20 | `ivascular essential pro 3` | REPEAT | the length is missing |

## Shelf positions - IMPORTANT

All positions above (`R1 / level 1 / column N`) come from `data/rak/posisi.csv`, which is **still EXAMPLE data** (not
confirmed by the team). Do not send the robot / TM to these coordinates until the real rack / level positions are confirmed;
see question 1 in `docs/CHECKLIST_DEMO.md`.
