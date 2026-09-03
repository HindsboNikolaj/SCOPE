#!/usr/bin/env bash
# Start a virtual display, wait until it actually answers, then run the command.
#
# The wait is not decoration. Xvfb returns from fork before the socket exists, so a Blender
# launched immediately after it fails with "cannot open display" perhaps one time in five.
# That kind of failure is easy to misread as a Blender problem.
set -euo pipefail

SCREEN="${XVFB_SCREEN:-1920x1080x24}"
DISPLAY_NUM="${DISPLAY:-:99}"
Xvfb "$DISPLAY_NUM" -screen 0 "$SCREEN" +extension GLX +extension RANDR +render -noreset \
     > /tmp/xvfb.log 2>&1 &
XVFB_PID=$!

for _ in $(seq 1 50); do
  if [ -e "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]; then break; fi
  sleep 0.2
done
if [ ! -e "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]; then
  echo "[entrypoint] Xvfb never came up:" >&2; cat /tmp/xvfb.log >&2; exit 1
fi

# A window manager, so Blender's window is mapped and actually painted. Without it
# `screen.screenshot_area` returns a black rectangle: the operator reads what is on screen,
# and with no window manager nothing ever gets put there.
WM_PID=""
if command -v openbox >/dev/null 2>&1 && [ "${SCOPE_WM:-1}" != "0" ]; then
  openbox > /tmp/openbox.log 2>&1 &
  WM_PID=$!
  sleep 1
fi

cleanup() {
  [ -n "$WM_PID" ] && kill "$WM_PID" 2>/dev/null || true
  kill "$XVFB_PID" 2>/dev/null || true
}
trap cleanup EXIT
exec "$@"
