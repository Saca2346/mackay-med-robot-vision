#!/bin/bash
# 10 Nov 2026: record the slide-1 video on the AI PC: a virtual screen (Xvfb :91, 1920x1080), an xterm on the left
# running live_start.sh, the program's live window on the right, ffmpeg x11grab at 15 fps, real speed.
cd ~/Downloads/claude_offline_20261101/aipc_20261104/tasks
export DISPLAY=:91
Xvfb :91 -screen 0 1920x1080x24 -nolisten tcp > xvfb_live.log 2>&1 &
XPID=$!
sleep 2
xsetroot -solid "#1D1E22"
ffmpeg -y -f x11grab -video_size 1920x1080 -framerate 15 -i :91 -c:v libx264 -preset veryfast -crf 21 \
  -pix_fmt yuv420p live1110.mp4 > ffmpeg_live.log 2>&1 &
FPID=$!
sleep 1
xterm -geometry 92x64+0+0 -fa "DejaVu Sans Mono" -fs 10 -bg "#181A20" -fg "#E6E6E6" -T "terminal" \
  -e bash live_start.sh
sleep 1
kill -INT $FPID; wait $FPID
kill $XPID
echo "recorded: $(ls -l live1110.mp4)"
