"""Check that a models folder works with the network switched off.

Run it from the repository root, with the folder given as an argument or in ANONYMIZER_MODELS:

    python3 examples/offline_check.py [MODELS_FOLDER]

The script makes every connection attempted through Python's socket module fail, then analyses the sample photo with
the Careful level, which loads every model. It writes nothing. The exit code is 0 when the analysis worked and no
connection was attempted, and 1 otherwise. A library that reaches the network without going through that module is not
covered.
"""

import os
import socket
import sys

import anonymizer

attempts = 0


def refuse(*args, **kwargs):
    global attempts
    attempts += 1
    raise OSError("the network is switched off for this check")


socket.socket.connect = refuse
socket.getaddrinfo = refuse

folder = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ANONYMIZER_MODELS")
try:
    with anonymizer.Anonymizer(anonymizer.Options(detection="careful"), models_dir=folder) as anon:
        analysis = anon.analyze(os.path.abspath("demo/street-westminster.jpg"))
except anonymizer.ModelLoadError as error:
    print("a model could not be loaded:", error.code)
    sys.exit(1)
print("analysed with the network off:", len(analysis.items), "items")
if attempts:
    print("a connection was attempted:", attempts, "times")
    sys.exit(1)
