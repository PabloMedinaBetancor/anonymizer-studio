# Getting started

This page installs Anonymizer Studio, fetches its detection models and takes one photo through the Studio. If you write code, do the first two steps, then go to [Photos in Python](tutorials/photos-in-python.md).

You need Python 3.10 or later, a network connection for the downloads and, for the PDF report, a Chrome, Edge or Chromium already on the computer. The Studio was run on Ubuntu 22.04 with Python 3.10, in Chrome and Firefox 157. **Windows, macOS and Safari were not tried**, and the files in `packaging/windows/` never ran on a Windows machine.

## Install

Clone the repository (`git clone https://github.com/PabloMedinaBetancor/anonymizer-studio.git`) and, from its folder, in a virtual environment:

```console
$ python3 -m venv .venv && .venv/bin/pip install -U pip
$ .venv/bin/pip install .
$ source .venv/bin/activate
```

Upgrade pip first, as above: the pip of Ubuntu 22.04 (22.0.2) stops with "Failed building wheel for UNKNOWN". On Ubuntu and Debian `python3 -m venv` stops with "ensurepip is not available" until you run `sudo apt install python3-venv`. Without `sudo`, install for your own user: `python3 -m pip install --user -U pip`, then `python3 -m pip install --user .`. The `anonymizer-studio` command is then on the `PATH` from your next login; until then start the Studio with `python3 -m anonymizer.ui`. `anonymizer-studio --version` prints the version and loads no model, so it is a quick check that the install worked.

`pip install ".[window]"` adds an optional window of its own for the Studio, instead of a browser tab.

## Fetch the models

```console
$ python3 scripts/fetch_models.py
```

The models are not in the package. The script downloads them from GitHub and ModelScope into `models/` and checks each against its SHA-256. It is the only time the product uses the network.

The models are looked for in this order: the folder given with `models_dir=` or `--models`, the folder named by `ANONYMIZER_MODELS`, `models/` in the current folder, and `models/` of the source tree. After `pip install .`, start from the checkout or set `ANONYMIZER_MODELS`. A missing or changed model is a `ModelLoadError` at the first analysis, and it names the folders it looked in.

### A computer with no network

Copy the whole `models/` folder (the `.onnx` files and `SHA256SUMS`) and point `ANONYMIZER_MODELS` at it. For the packages, build wheels on a connected computer with the same operating system, architecture and Python version, copy the folder, and install from it:

```console
$ python3 -m pip wheel -w wheels .
$ python3 -m pip install --no-index --find-links wheels anonymizer-studio
```

`python3 examples/offline_check.py FOLDER` then analyses the sample photo with every network connection refused, and exits with 0 when that works. These steps ran on Ubuntu 22.04, not on a computer that was physically offline.

## Blur a first photo

```console
$ anonymizer-studio
```

It prints an address and opens it in your browser. Open the address it prints, not a bookmark: the address carries the session. The Studio stops by itself after ten minutes without a page open.

1. **Import.** Press **Import** and choose `demo/street-westminster.jpg`, a photo of a pedestrian crossing in London (credit on [Licence and credits](licence-and-credits.md)). Leave the settings as they are and press **Analyze 1 clip**.
2. **Analyze.** The clip shows its progress. Nothing is written yet. When it is analysed, click its card.
3. **Review.** The items are drawn on the picture, and the ones the detectors are not sure about are badged **Not sure**. `N` jumps to the next one and `B` blurs or unblurs it. **Add missed area** covers a face or a plate that has no box.
4. **Export.** Press **Render 1 clip**. The photo and its reports go to a folder named `anonymized` next to the original. A file is never replaced: the Studio adds `-2` to a name that is taken.

Before you publish, look at the result in **Anonymized**: every face and plate you can see should be blurred, and the **Not sure** items left visible should be neither. A face or a plate the detectors did not find has no box and is not listed.

Next: [Your first video](tutorials/first-video.md) takes a clip through the same screens, with pictures.
