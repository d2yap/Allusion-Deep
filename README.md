# Allusion+Deep

This Allusion fork integrates local image taggers for image sorting. The app can use a WD14-compatible helper to tag images automatically.

<img alt="Allusion" src="./resources/images/helpcenter/logo-about-helpcenter-dark.jpg" width="250" />

Allusion is a tool built for artists, aimed to help you organize your **Visual Library** – A single place that contains your entire collection of references, inspiration and any other kinds of images.

[Read more about Allusion →](https://allusion-app.github.io/)

## Installation

Find the latest version of Allusion on the [Releases](https://github.com/allusion-app/Allusion/releases) page.
The application can be automatically updated when a new version becomes available.

## Development

### Quick Start

You need to have [NodeJS](https://nodejs.org/en/download/) and a package manager such as [Yarn](https://yarnpkg.com/lang/en/docs/install/) installed.
Then run the following commands to get started:

### Requirements

- pip
- Python3

  > You will get an ENOENT error if you do not have these installed.

### Additional (WD14 tagger helper)

-- Python packages required to run `resources/wd14/tagger.py` (use `pip`):

```bash
python -m pip install pillow numpy pandas onnxruntime scipy torch safetensors
```

- The helper accepts tag lists in `csv` files with `name` and `category` columns, as well as plain `txt` or `json` files.

### GPU tagging (recommended for large libraries)

The app keeps one tagger process alive and reuses the loaded model, so tagging costs
~0.25 s/image on a GPU instead of ~2.3 s/image when spawning Python per image.
Without a GPU environment it still runs on CPU (~0.55 s/image).

1. Run `yarn setup:gpu` (or `bash scripts/setup-gpu-tagger.sh`). This creates a
   Python 3.13 environment at `~/venvs/allusion-gpu` with `onnxruntime-gpu` and the
   CUDA 13 / cuDNN 9 runtime libraries it needs. No system CUDA install is required.
2. Pick an ONNX model (e.g. `model.onnx`) and its `selected_tags.csv` in
   **Settings → Advanced**.

The app uses `~/venvs/allusion-gpu/bin/python` when it exists and silently falls back
to CPU otherwise. Overrides:

| Variable | Purpose |
| --- | --- |
| `DEEPDANBOORU_CUDA_PYTHON` | Use a different Python interpreter for GPU tagging |
| `DEEPDANBOORU_PYTHON` | Force a specific interpreter and CPU-only execution |

The same helper can tag a folder from the command line, without the app:

```bash
# One image (also works for `--image dir/` or a glob)
resources/wd14/tagger.py --image photo.png --model model.onnx --taglist selected_tags.csv
# Many images, emitted as JSON lines; the model is loaded once
python resources/wd14/tagger.py --batch ./images --batch-size 8 --device auto \
  --model model.onnx --taglist selected_tags.csv
```

1. Run `yarn install` to install or update all necessary dependencies.
2. Run `yarn dev` to build the project files to the `/build` directory. This will keep running to immediately build changed files when they are updated.
3. In a second terminal, run `yarn start` to start the application. Refresh the window (Ctrl/Cmd + R) after modifying a file to load the updated build files.

### Release Build

An installable executable can be built using `yarn package` for your platform in the `/dist` folder. The building is performed using the [electron-builder](https://www.electron.build/) package, and is configured by a section in the `package.json` file.
Builds are automatically published to Github Releases when a tag is created in GitHub.

## Further Information

Anything related from documentation to frequently asked questions can be found in our [wiki](https://github.com/allusion-app/Allusion/wiki).
