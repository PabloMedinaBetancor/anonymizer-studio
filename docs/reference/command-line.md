# Command line

The command that starts the Studio, the scripts that fetch the detection models, and the environment variables the product reads. Every option below was run once against version 2.2.0.

## `python3 -m anonymizer.ui`

Starts the Studio, the local web application, on this computer only. `anonymizer-studio` is the same command, installed as a console script.

```console
$ python3 -m anonymizer.ui [--version] [--demo] [--port PORT] [--no-browser] [--models MODELS] [--engine {process,thread}]
```

| Option | Default | What it does |
|---|---|---|
| `--version` | | Prints `Anonymizer Studio 2.2.0` and ends with exit code 0. |
| `--port PORT` | a free port | The port to listen on, from 0 to 65535; 0 takes a free one. |
| `--no-browser` | off | Prints the address and does not open it in your browser, or in the optional window. |
| `--models MODELS` | see below | The models folder. Without it: `ANONYMIZER_MODELS`, then `./models`, then the `models` folder of the source tree. |
| `--engine {process,thread}` | `process` | `process` runs the engine in a worker process that can be killed, for Stop and the silence limit; `thread` is for tests. |
| `--demo` | off | Demo mode, below. |
| `-h`, `--help` | | Prints the options and ends. |

It prints the address, `Anonymizer Studio: http://127.0.0.1:<port>/?t=<token>`, with a token that changes at every start, and the stopping rule. If no model file is where it will look, a third line on standard error says where it looked.

It ends with exit code 0 on Ctrl+C, `SIGTERM` and `SIGHUP` (a closed terminal, `kill`), the idle stop and the end of its own window, and deletes its work folder in each case. It ends with exit code 2 when the port is taken or an option has a bad value. It listens on `127.0.0.1` only ([Privacy](../concepts/privacy.md)).

**Demo mode** (`--demo`) shows the Studio without keeping anything: no settings file, files uploaded and not browsed, results as downloads. The limits are 10 photos, 10 clips in all, 20 seconds and 720p per video, and 100 MB per upload, and the deep level is refused. One process is one session: start one process for each person. Behind a reverse proxy that ends TLS, the proxy must send `Host: 127.0.0.1:<port>` to the Studio, and `X-Forwarded-Proto: https` makes the cookie `Secure`.

## `scripts/fetch_models.py`

Downloads the six detection models from GitHub and ModelScope and checks each against `models/SHA256SUMS`. This is the only network use of the product, and it happens once. It uses the standard library only.

```console
$ python3 scripts/fetch_models.py [--models MODELS] [--only NAME [NAME ...]]
```

| Option | Default | What it does |
|---|---|---|
| `--models MODELS` | the `models/` folder of the checkout | The folder for the `.onnx` files. After fetching into another folder, set `ANONYMIZER_MODELS` to it, or pass `--models` to the Studio. |
| `--only NAME [NAME ...]` | every model | Fetch only these files. A name that is not in `SHA256SUMS` is refused. |
| `-h`, `--help` | | Prints the options and ends. |

A file that is already there and matches its SHA-256 is left alone, and a download that does not match is discarded. The script ends with 0 when every wanted model is there and matches, and with 1 when a download failed or did not match: it says which, and a second run fetches only the missing models.

`scripts/fetch_reference_plate_model.py [--models MODELS]` downloads the v1 plate model that only `Options(plate_path="yolo")` and the benchmark's plate comparison need. The product does not ship it, because its licence is not settled ([Licence and credits](../licence-and-credits.md)); `--help` prints the usage without downloading anything.

## Environment variables

| Variable | Read by | What it does |
|---|---|---|
| `ANONYMIZER_MODELS` | the Python API, the Studio | The models folder, when `models_dir` and `--models` are not given. |
| `XDG_CONFIG_HOME` | the Studio | The folder that holds `anonymizer-studio/settings.json`; `~/.config` when it is not set. |
| `APPDATA` | the Studio on Windows | The folder that holds `AnonymizerStudio\settings.json`. |
| `OPENCV_LOG_LEVEL` | OpenCV | `import anonymizer` sets it to `ERROR` unless you have set it. |
| `DISPLAY`, `WAYLAND_DISPLAY` | the Studio | Whether there is a screen for the window or the browser. |
| `PROGRAMFILES`, `LOCALAPPDATA` | the Studio on Windows | Where to look for Chrome or Edge, which print the PDF report. |
