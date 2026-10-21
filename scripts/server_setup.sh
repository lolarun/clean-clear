#!/usr/bin/env bash
# Prepare a rented GPU server (Ubuntu, root) to work for the controller: system packages, the tool, the worker as a
# supervisor service. Run in the tool folder:  bash scripts/server_setup.sh SERVER_ID [WORK_DIR]
# Before: the rclone configuration with the remote "gdrive" (rooted at the project's Drive folder) must be in
# ~/.config/rclone/rclone.conf. Safe to run again: finished steps are skipped.
set -euo pipefail
SERVER_ID=${1:?usage: server_setup.sh SERVER_ID [WORK_DIR]}
WORK=${2:-/workspace}
TOOL=$(cd "$(dirname "$0")/.." && pwd)
echo "== setup $SERVER_ID: tool $TOOL, work $WORK"

need=()
for c in ffmpeg rclone supervisorctl; do command -v "$c" >/dev/null || need+=("$c"); done
python3 -c "import venv, ensurepip" 2>/dev/null || need+=(python3-venv)
if [ ${#need[@]} -gt 0 ]; then
  pk=()
  for c in "${need[@]}"; do case $c in supervisorctl) pk+=(supervisor);; *) pk+=("$c");; esac; done
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "${pk[@]}"
fi

if [ ! -x "$TOOL/.venv/bin/clean-clear" ]; then
  bash "$TOOL/scripts/install.sh"
fi
"$TOOL/.venv/bin/clean-clear" --version

rclone lsf gdrive: --max-depth 1 >/dev/null || { echo "rclone remote 'gdrive' does not work: copy rclone.conf first"; exit 1; }
chmod 600 ~/.config/rclone/rclone.conf

mkdir -p "$WORK"
if [ ! -f "$WORK/worker.json" ]; then
  cat > "$WORK/worker.json" <<EOF
{
 "server_id": "$SERVER_ID",
 "tool": "$TOOL",
 "work": "$WORK",
 "remote": "gdrive:",
 "staging": "staging",
 "log": "log",
 "heartbeat_seconds": 300,
 "health_clip_seconds": 180
}
EOF
fi

cat > /etc/supervisor/conf.d/video-worker.conf <<EOF
[program:video-worker]
command=$TOOL/.venv/bin/python $TOOL/scripts/worker.py $WORK/worker.json
directory=$WORK
autostart=true
autorestart=true
stopwaitsecs=60
redirect_stderr=true
stdout_logfile=$WORK/worker.out.log
EOF
pgrep -x supervisord >/dev/null || supervisord -c /etc/supervisor/supervisord.conf
supervisorctl reread && supervisorctl update
supervisorctl start video-worker || true
supervisorctl status video-worker
echo "== $SERVER_ID ready; the worker writes $WORK/status.txt and gdrive:log/servers/$SERVER_ID/heartbeat.json"
