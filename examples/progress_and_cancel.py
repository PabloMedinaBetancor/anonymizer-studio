# docs-test: real-only - it cancels a real analysis from another thread, which needs a clip that takes time
"""Show the events of a job and stop one from another thread.

Run it from the repository root:

    python3 examples/progress_and_cancel.py [CLIP [OUTPUT]]

CLIP defaults to examples/data/street-madrid-5s.mp4 and OUTPUT to stopped. The script analyses the clip, asks a
timer thread to stop it after three seconds, and shows that the stop arrives as the event Finished("stopped")
and then as the exception Cancelled, and that nothing is written under OUTPUT. It writes no media file.
"""

import os
import sys
import threading

import anonymizer

clip = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "examples/data/street-madrid-5s.mp4")
output = sys.argv[2] if len(sys.argv) > 2 else "stopped"
os.makedirs(output, exist_ok=True)

events = []


def on_event(event):
    name = type(event).__name__
    if isinstance(event, anonymizer.Finished):
        name = f"Finished({event.status})"
    if not events or events[-1] != name:
        events.append(name)


token = anonymizer.CancelToken()
threading.Timer(3.0, token.cancel).start()

try:
    anonymizer.anonymize_video(clip, os.path.join(output, "never.mp4"), on_event=on_event, cancel=token)
except anonymizer.Cancelled:
    events.append("Cancelled raised")

print("events:", ", ".join(events))
print("files in", output + ":", os.listdir(output))
