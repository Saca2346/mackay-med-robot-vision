#!/usr/bin/env bash
# Uji kamera 16 kotak di SERVER (jalankan di terminal desktop Remmina), webcam dari laptop lewat tunnel SSH.
# Prasyarat: di laptop, stream_webcam.py dan tunnel SSH sudah jalan (lihat docs/archive_id/PANDUAN_UJI_KAMERA.md langkah 2-3).
#
#   bash scripts/jalankan_uji_kamera_server.sh            # 16 kotak, 90 detik per kotak
#   DEVICE=cpu bash scripts/jalankan_uji_kamera_server.sh  # jika muncul CUDA out of memory
#   TUNGGU=300 bash scripts/jalankan_uji_kamera_server.sh  # tunggu tunnel lebih lama
#   WAKTU=60 bash scripts/jalankan_uji_kamera_server.sh   # ganti waktu per kotak
#   RENCANA=results/rencana_lain.txt NAMA=ulang1 bash scripts/jalankan_uji_kamera_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."

WAKTU=${WAKTU:-90}
RENCANA=${RENCANA:-results/rencana_uji_kamera_16.txt}
NAMA=${NAMA:-$(date +%Y%m%d_%H%M)}
SUMBER=${SUMBER:-http://127.0.0.1:18090/video.mjpg}
CONFIG=${CONFIG:-config/box_pipeline_config_server.yaml}
LOG="results/uji_kamera_${NAMA}.csv"
TUNGGU=${TUNGGU:-180}   # detik menunggu tunnel dari laptop
DEVICE_ARG=()
[ -n "${DEVICE:-}" ] && DEVICE_ARG=(--device "$DEVICE")   # DEVICE=cpu bila VRAM GPU penuh (training lain berjalan)

source "${VENV:-$HOME/trse_venv}/bin/activate"

echo "== cek 1/3: model dan data =="
[ -f models/box_v4s.pt ] || { echo "GAGAL: models/box_v4s.pt tidak ada"; exit 1; }
[ -f data/rak/inventaris.csv ] || { echo "GAGAL: data/rak/inventaris.csv tidak ada (jalankan dari folder proyek)"; exit 1; }
[ -f "$RENCANA" ] || { echo "GAGAL: rencana $RENCANA tidak ada"; exit 1; }

echo "== cek 2/3: uji teks otomatis (harus 20 lulus) =="
python scripts/uji_checklist.py | tail -2

echo "== cek 3/3: stream dari laptop ($SUMBER), menunggu sampai ${TUNGGU}s =="
t0=$(date +%s)
# stream MJPEG tidak pernah berakhir: curl selalu habis waktu (exit 28), jadi yang dicek kode HTTP-nya saja
stream_ok() { [ "$(curl -s --max-time 3 -o /dev/null -w '%{http_code}' "$SUMBER" || true)" = "200" ]; }
until stream_ok; do
  if [ $(( $(date +%s) - t0 )) -ge "$TUNGGU" ]; then
    echo "GAGAL: stream tidak terjangkau setelah ${TUNGGU}s. Di laptop pastikan 2 jendela cmd terbuka:"
    echo "  1) venv\\Scripts\\python.exe scripts\\stream_webcam.py   (webcam eksternal terpasang, program kamera lain ditutup)"
    echo "  2) ssh -N -R 18090:127.0.0.1:8090 Zen@140.113.149.94   (password sudah diisi, jendela tidak ditutup)"
    exit 1
  fi
  echo "  menunggu tunnel/stream dari laptop... ($(( $(date +%s) - t0 ))s)"
  sleep 3
done
echo "== cek gambar kamera =="
mkdir -p results
set +e
python scripts/cek_stream.py "$SUMBER" --simpan "results/frame_cek_${NAMA}.jpg"
rc=$?
set -e
if [ "$rc" = 1 ]; then exit 1; fi
if [ "$rc" = 2 ]; then
  read -r -p "Ada peringatan gambar. Lanjut uji? [y/N] " jawab
  [ "${jawab:-n}" = "y" ] || { echo "Dihentikan. Perbaiki kamera lalu jalankan ulang."; exit 1; }
fi

mkdir -p results
echo
echo "Mulai: $(grep -c '^[^#!]' "$RENCANA") kotak x ${WAKTU} detik. Ikuti urutan di jendela video."
echo "Tombol di jendela video: q = selesai | s = simpan frame | r = ulang permintaan yang sama"
echo "Hasil: $LOG | rekaman: results/uji_kamera_${NAMA}.mp4"
echo

python scripts/test_webcam_box.py \
  --config "$CONFIG" --source "$SUMBER" "${DEVICE_ARG[@]}" \
  --rencana "$RENCANA" --waktu-per-barang "$WAKTU" --expected 20 \
  --inventaris data/rak/inventaris.csv \
  --log-stress "$LOG" \
  --debug-readings "results/bacaan_uji_kamera_${NAMA}.csv" \
  --record "results/uji_kamera_${NAMA}.mp4" ${EXTRA:-}

echo
echo "== ringkasan =="
python scripts/ringkas_uji_kamera.py "$LOG"
