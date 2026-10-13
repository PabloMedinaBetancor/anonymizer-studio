"""Imported first by the package: OpenCV 5 prints "[ WARN:0@0.1] global net_impl_backend.cpp:345 setPreferableTarget Targets are not
supported by the new graph engine for now" on stderr each time a model is loaded, which reads as a fault in the first example of the
README. It is not one (the models run on the CPU, as meant), so OpenCV's log goes down to errors, unless the person has chosen a level."""

import os

LEVEL = os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # read when OpenCV is first imported
