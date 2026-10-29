# https://huggingface.co/spaces/SmilingWolf/wd-tagger

import argparse
import ctypes
import glob
import json
import os
import sys
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

DEFAULT_MODEL_FILENAMES = ("model.onnx", "model.pt", "model.pth", "model.h5")
DEFAULT_LABEL_FILENAMES = ("selected_tags.csv", "tags.csv", "tags.txt", "names.txt")

_CUDA_LIB_HANDLES: List = []


def _physical_core_count() -> int:
    """Count unique (physical id, core id) pairs; SMT siblings slow ORT down."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            pairs = set()
            phys = core = None
            for line in handle:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
            if pairs:
                return len(pairs)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def build_session_options(threads: int = 0) -> "rt.SessionOptions":
    options = rt.SessionOptions()
    options.log_severity_level = 3
    options.intra_op_num_threads = threads or _physical_core_count()
    options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def activate_venv_cuda_libs() -> List[str]:
    """When running from a venv with pip CUDA wheels, ORT needs their directories on the loader path.

    Setting LD_LIBRARY_PATH after process start has no effect on dlopen, so call
    dlopen ourselves and keep the handles alive for the life of the process.
    """
    sites = {os.path.dirname(os.path.dirname(rt.__file__))}
    try:
        import importlib.metadata as metadata

        for dist_name in ("nvidia-cublas", "nvidia-cudnn-cu13", "nvidia-cuda-runtime"):
            try:
                for entry in metadata.files(dist_name) or []:
                    if entry.name.startswith("lib") and ".so" in entry.name:
                        sites.add(str(entry.locate().parent))
            except metadata.PackageNotFoundError:
                continue
    except ImportError:
        pass

    loaded = []
    for site_dir in sites:
        nvidia_dir = os.path.join(site_dir, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue
        for entry in sorted(os.listdir(nvidia_dir)):
            lib_dir = os.path.join(nvidia_dir, entry, "lib")
            if not os.path.isdir(lib_dir):
                continue
            for path in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                try:
                    loaded.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
                except OSError:
                    pass
    return loaded


def create_session(model_path: str, device: str = "auto") -> "rt.InferenceSession":
    """Create an inference session, preferring the GPU unless device == 'cpu'."""
    options = build_session_options()
    if device != "cpu":
        global _CUDA_LIB_HANDLES
        _CUDA_LIB_HANDLES = activate_venv_cuda_libs()
    if device != "cpu" and "CUDAExecutionProvider" in rt.get_available_providers():
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if device == "cuda":
            print(
                "WD14 worker: CUDAExecutionProvider not available in this onnxruntime build",
                file=sys.stderr,
                flush=True,
            )
        wanted = ["CPUExecutionProvider"]
    session = rt.InferenceSession(model_path, options, providers=wanted)
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        print(
            "WD14 worker: CUDAExecutionProvider failed to load, fell back to CPU",
            file=sys.stderr,
            flush=True,
        )
    return session

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-slider-step", type=float, default=0.05)
    parser.add_argument("--score-general-threshold", type=float, default=0.35)
    parser.add_argument("--score-character-threshold", type=float, default=0.85)
    return parser.parse_args()


def load_labels(dataframe) -> Tuple[List[str], List[int], List[int], List[int]]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def load_labels_from_path(path: str) -> Tuple[List[str], List[int], List[int], List[int]]:
    lower = path.lower()
    if lower.endswith(".csv"):
        dataframe = pd.read_csv(path)
        if "name" not in dataframe.columns:
            if "tag" in dataframe.columns:
                dataframe = dataframe.rename(columns={"tag": "name"})
            elif "label" in dataframe.columns:
                dataframe = dataframe.rename(columns={"label": "name"})
        if "category" not in dataframe.columns:
            dataframe["category"] = 0
        return load_labels(dataframe)

    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file_handle:
            entries = json.load(file_handle)
        names = []
        for entry in entries:
            if isinstance(entry, str):
                names.append(entry)
            elif isinstance(entry, dict):
                names.append(str(entry.get("name") or entry.get("tag") or entry.get("label") or ""))
        dataframe = pd.DataFrame({"name": names, "category": [0] * len(names)})
        return load_labels(dataframe)

    with open(path, "r", encoding="utf-8") as file_handle:
        names = [line.strip() for line in file_handle if line.strip()]
    dataframe = pd.DataFrame({"name": names, "category": [0] * len(names)})
    return load_labels(dataframe)


def resolve_first_existing_file(paths: Iterable[str]) -> str:
    for candidate in paths:
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


def resolve_model_and_taglist(
    project_path: Optional[str],
    model_path: Optional[str],
    taglist_path: Optional[str],
) -> Tuple[str, str]:
    if model_path and taglist_path:
        return model_path, taglist_path

    if not project_path:
        return model_path or "", taglist_path or ""

    resolved_model = resolve_first_existing_file(
        os.path.join(project_path, filename) for filename in DEFAULT_MODEL_FILENAMES
    )
    resolved_taglist = resolve_first_existing_file(
        os.path.join(project_path, filename) for filename in DEFAULT_LABEL_FILENAMES
    )
    return model_path or resolved_model, taglist_path or resolved_taglist


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.last_loaded_repo = None
        self.model = None

    def download_model(self, model_repo):
        raise RuntimeError("download_model removed: this Predictor only supports local files")

    def load_model_from_files(self, model_path: str, csv_path: str):
        # Load taglist CSV and ONNX model from local paths only.
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Taglist not found: {csv_path}")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        key = f"{model_path}:{csv_path}"
        if key == self.last_loaded_repo:
            return

        sep_tags = load_labels_from_path(csv_path)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        del self.model
        model = create_session(model_path)
        try:
            _, height, width, _ = model.get_inputs()[0].shape
            self.model_target_size = height
        except Exception:
            self.model_target_size = 512

        self.last_loaded_repo = key
        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def _run_scores(self, batch):
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        return self.model.run([label_name], {input_name: batch})[0]

    def _decode_predictions(
        self,
        preds,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        labels = list(zip(self.tag_names, preds.astype(float)))
        truncated = {}
        scores = {}

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        for tag, score in general_names:
            truncated[tag] = float(score)
            if tag in general_res:
                scores[tag] = float(score)
        for tag, score in character_names:
            truncated[tag] = float(score)
            if tag in character_res:
                scores[tag] = float(score)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = ", ".join(sorted_general_strings)

        payload = {
            "tags": build_tags_payload(general_res, character_res),
            "general_string": sorted_general_strings,
            "rating": rating,
            "characters": character_res,
            "general": general_res,
            # Top scores regardless of threshold, for callers doing their own filtering
            "scores": dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)),
            "truncated": dict(sorted(truncated.items(), key=lambda item: item[1], reverse=True)),
        }
        return payload

    def predict_one(
        self,
        image_path,
        model_path,
        taglist_path,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        self.load_model_from_files(model_path, taglist_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGBA")
        batch = self.prepare_image(image)
        preds = self._run_scores(batch)
        return self._decode_predictions(
            preds[0], general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled
        )

    def predict_batch(
        self,
        image_paths,
        model_path,
        taglist_path,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        """Tag several images in one forward pass. Returns (payload, failures)."""
        self.load_model_from_files(model_path, taglist_path)
        results = {}
        failures = {}
        prepared_paths = []
        prepared = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as handle:
                    image = handle.convert("RGBA")
                prepared.append(self.prepare_image(image))
                prepared_paths.append(image_path)
            except Exception as err:  # unreadable/corrupt image: report, keep going
                failures[image_path] = str(err)

        if prepared:
            batch = np.concatenate(prepared, axis=0)
            del prepared
            preds = self._run_scores(batch)
            for index, image_path in enumerate(prepared_paths):
                results[image_path] = self._decode_predictions(
                    preds[index],
                    general_thresh,
                    general_mcut_enabled,
                    character_thresh,
                    character_mcut_enabled,
                )
        return results, failures

    def predict(
        self,
        image,
        model_path,
        taglist_path,
        general_thresh,
        general_mcut_enabled,
        character_thresh,
        character_mcut_enabled,
    ):
        self.load_model_from_files(model_path, taglist_path)
        preds = self._run_scores(self.prepare_image(image))[0]
        payload = self._decode_predictions(
            preds,
            general_thresh,
            general_mcut_enabled,
            character_thresh,
            character_mcut_enabled,
        )
        return payload["general_string"], payload["rating"], payload["characters"], payload["general"]


def build_tags_payload(general_res, character_res):
    merged = {}
    merged.update(general_res)
    merged.update(character_res)
    return [
        {"tag": tag_name, "score": float(score)}
        for tag_name, score in sorted(merged.items(), key=lambda item: item[1], reverse=True)
    ]


def expand_image_args(patterns: Iterable[str]) -> List[str]:
    """Expand CLI args: a directory means its immediate images, globs are expanded too."""
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
    found = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            entries = sorted(
                entry
                for entry in glob.glob(os.path.join(pattern, '*'))
                if os.path.splitext(entry)[1].lower() in image_exts
            )
        elif any(char in pattern for char in '*?['):
            entries = sorted(glob.glob(pattern))
        else:
            entries = [pattern]
        found.extend(entry for entry in entries if os.path.isfile(entry))
    return found


def serve(args, model_path: str, taglist_path: str) -> None:
    """Long-lived worker: read JSON requests (one per line) from stdin, answer on stdout.

    Every response is a single line so the caller can pair replies with requests.
    """
    predictor = Predictor()
    if args.prewarm:
        try:
            predictor.load_model_from_files(model_path, taglist_path)
        except Exception as err:
            print(json.dumps({"ready": False, "error": str(err)}, ensure_ascii=False), flush=True)
            return
    print(
        json.dumps(
            {
                "ready": True,
                "device": predictor.model.get_providers()[0],
                "providers": predictor.model.get_providers(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id = None
        started = time.perf_counter()
        try:
            request = json.loads(line)
            request_id = request.get("id")
            if request.get("op") == "ping":
                response = {"id": request_id, "ready": True}
            elif request.get("op") == "shutdown":
                break
            elif request.get("op") == "predict":
                paths = request.get("images") or []
                if isinstance(paths, str):
                    paths = [paths]
                general_thresh = request.get("general_thresh", args.general_thresh)
                character_thresh = request.get("character_thresh", args.character_thresh)
                general_mcut = request.get("general_mcut", args.general_mcut)
                character_mcut = request.get("character_mcut", args.character_mcut)
                if len(paths) == 1:
                    payload = predictor.predict_one(
                        paths[0],
                        model_path,
                        taglist_path,
                        general_thresh,
                        general_mcut,
                        character_thresh,
                        character_mcut,
                    )
                    response = {"id": request_id, "results": {paths[0]: payload}}
                else:
                    results, failures = predictor.predict_batch(
                        paths,
                        model_path,
                        taglist_path,
                        general_thresh,
                        general_mcut,
                        character_thresh,
                        character_mcut,
                    )
                    response = {"id": request_id, "results": results, "failures": failures}
            else:
                response = {"id": request_id, "error": f"Unknown op: {request.get('op')}"}
        except Exception as err:
            response = {"id": request_id, "error": str(err)}
        response["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        print(json.dumps(response, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to input image (or use --serve / --batch)')
    parser.add_argument('--batch', nargs='+', help='Tag several images/directories/globs in one pass')
    parser.add_argument('--serve', action='store_true', help='Run as a long-lived JSON-lines worker on stdin/stdout')
    parser.add_argument('--prewarm', action='store_true', help='With --serve, load the model before announcing readiness')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cpu',
                        help="Execution provider: 'auto' tries CUDA then falls back to CPU, 'cpu' forces CPU")
    parser.add_argument('--project', help='Path to a local WD14 project directory')
    parser.add_argument('--model', help='Path to ONNX or Torch model file')
    parser.add_argument('--taglist', help='Path to taglist CSV, JSON, or text file')
    parser.add_argument('--threshold', type=float, default=None, help='Apply one threshold to both general and character tags')
    parser.add_argument('--general-thresh', type=float, default=0.35)
    parser.add_argument('--character-thresh', type=float, default=0.85)
    parser.add_argument('--general-mcut', action='store_true')
    parser.add_argument('--character-mcut', action='store_true')
    parser.add_argument('--batch-size', type=int, default=8, help='Images per forward pass for --batch/--serve')
    args = parser.parse_args()

    model_path, taglist_path = resolve_model_and_taglist(args.project, args.model, args.taglist)
    if not model_path:
        print(json.dumps({"tags": [], "error": "Model file not found. Pass --model or --project."}, ensure_ascii=False))
        return
    if not taglist_path:
        print(json.dumps({"tags": [], "error": "Tag list file not found. Pass --taglist or --project."}, ensure_ascii=False))
        return

    if args.threshold is not None:
        args.general_thresh = args.threshold
        args.character_thresh = args.threshold

    if args.serve:
        serve(args, model_path, taglist_path)
        return

    paths = expand_image_args((args.batch or []) + ([args.image] if args.image else []))
    if not paths:
        print(json.dumps({"tags": [], "error": "No input images found."}, ensure_ascii=False))
        return

    predictor = Predictor()
    if len(paths) == 1:
        out = predictor.predict_one(
            paths[0],
            model_path,
            taglist_path,
            args.general_thresh,
            args.general_mcut,
            args.character_thresh,
            args.character_mcut,
        )
        print(json.dumps(out, ensure_ascii=False))
        return

    # Batch mode: one JSON object per line, tagged in chunks of --batch-size
    size = max(1, args.batch_size)
    for index in range(0, len(paths), size):
        chunk = paths[index:index + size]
        results, failures = predictor.predict_batch(
            chunk,
            model_path,
            taglist_path,
            args.general_thresh,
            args.general_mcut,
            args.character_thresh,
            args.character_mcut,
        )
        for path in chunk:
            if path in results:
                print(json.dumps({"image": path, **results[path]}, ensure_ascii=False), flush=True)
            else:
                print(json.dumps({"image": path, "tags": [], "error": failures.get(path, "unknown")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
