"""Recognition-only RapidOCR adapter with explicit ONNX Runtime providers.

Ported from apex-highlight-clipper-codex/gpu_ocr_backend.py, simplified for the
plugin: provider selection is strict (`cpu` / `dml` / `cuda`); an unavailable
provider raises instead of silently running the model on CPU. Session creation
goes through the pinned SDK (`timepin_vision.model.load_onnx_session`) so plugins
never import onnxruntime session plumbing themselves.

Interfaces (coordinated with the outcome/weapon agents):
- create_ocr_engine(backend='cpu', model_path=None, rec_batch_num=24)
    -> SimpleNamespace(text_rec=<RapidOCR TextRecognizer>, backend_info=dict)
       text_rec(crops) -> ([(text, confidence), ...], elapsed)
- create_full_ocr_engine(backend='cpu', **rapidocr_kwargs)
    -> RapidOCR instance callable as engine(image, use_cls=False, ...);
       detection/classification stay on CPU, only recognition uses `backend`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

_BACKEND_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}
# DirectML sessions forbid concurrent Run on one session; serialize device work.
_DML_DEVICE_LOCK = threading.RLock()


def _available_providers():
    from onnxruntime import get_available_providers

    return list(get_available_providers())


def _explicit_session(model_path, backend, device_id=0):
    """Create an InferenceSession that MUST run on the requested provider.

    Raises FileNotFoundError for a missing model, ValueError for an unknown
    backend, RuntimeError when the provider is unavailable or ORT fell back.
    """
    from timepin_vision.model import load_onnx_session

    provider = _BACKEND_PROVIDERS.get(backend)
    if provider is None:
        raise ValueError(f"未知 OCR 后端: {backend}（支持 cpu/dml/cuda）")
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OCR 模型不存在: {path}")
    available = _available_providers()
    if provider not in available:
        raise RuntimeError(
            f"{provider} 不可用；当前可用 provider: {available}。"
            f"请检查 onnxruntime 安装（dml 需 onnxruntime-directml，cuda 需 onnxruntime-gpu + 匹配的 CUDA/cuDNN）。"
        )
    providers = [(provider, {"device_id": str(device_id)})]
    if backend != "cpu":
        # Single ops may still run on CPU; the provider itself is enforced below.
        providers.append("CPUExecutionProvider")
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_mem_pattern = backend != "dml"
    session = load_onnx_session(str(path), providers=providers, sess_options=options)
    session.disable_fallback()
    if session.get_providers()[0] != provider:
        raise RuntimeError(
            f"{provider} 会话初始化回退到了 {session.get_providers()}；"
            f"未请求的后端不会用于 OCR。"
        )
    return session


class _RecognizeSession:
    """Adapter exposing RapidOCR's OrtInferSession surface on an SDK session."""

    def __init__(self, model_path, backend, device_id=0):
        self.session = _explicit_session(model_path, backend, device_id)
        self._lock = threading.Lock()
        self.backend = backend
        self.model_path = str(Path(model_path).resolve())

    def __call__(self, input_content):
        execution_lock = (
            _DML_DEVICE_LOCK if self.backend == "dml" else threading.RLock()
        )
        with execution_lock, self._lock:
            return self.session.run(
                self.get_output_names(),
                dict(zip(self.get_input_names(), [input_content])),
            )

    def get_input_names(self):
        return [v.name for v in self.session.get_inputs()]

    def get_output_names(self):
        return [v.name for v in self.session.get_outputs()]

    def get_character_list(self, key="character"):
        meta = self.session.get_modelmeta().custom_metadata_map
        return meta[key].splitlines()

    def describe(self):
        options = self.session.get_session_options()
        return {
            "backend": self.backend,
            "model_path": self.model_path,
            "providers": self.session.get_providers(),
            "provider_options": self.session.get_provider_options(),
            "memory_pattern": options.enable_mem_pattern,
            "execution_mode": str(options.execution_mode),
            "intra_op_threads": options.intra_op_num_threads,
            "inter_op_threads": options.inter_op_num_threads,
        }


def _default_rec_model():
    import rapidocr_onnxruntime

    return (
        Path(rapidocr_onnxruntime.__file__).parent
        / "models"
        / "ch_PP-OCRv4_rec_infer.onnx"
    )


def create_ocr_engine(*, backend="cpu", model_path=None, rec_batch_num=24, device_id=0):
    """Return .text_rec(crops) -> ([(text, confidence), ...], elapsed).

    CPU is the safe default. GPU requests are strict: an unavailable provider
    raises instead of silently running the whole model on CPU. Unsupported
    individual GPU operators may run on CPU. Existing OCR crop sizes,
    normalization, sorting, CTC and dictionary are preserved by reusing
    RapidOCR's TextRecognizer methods.
    """
    from rapidocr_onnxruntime.ch_ppocr_rec.text_recognize import TextRecognizer
    from rapidocr_onnxruntime.ch_ppocr_rec.utils import CTCLabelDecode

    if int(rec_batch_num) < 1:
        raise ValueError("rec_batch_num must be positive")
    if model_path is None:
        model_path = _default_rec_model()
    session = _RecognizeSession(model_path, backend, device_id)
    recognizer = TextRecognizer.__new__(TextRecognizer)
    recognizer.session = session
    recognizer.postprocess_op = CTCLabelDecode(character=session.get_character_list())
    recognizer.rec_batch_num = int(rec_batch_num)
    recognizer.rec_image_shape = [3, 48, 320]
    return SimpleNamespace(
        text_rec=recognizer,
        backend_info={**session.describe(), "rec_batch_num": recognizer.rec_batch_num},
    )


def create_full_ocr_engine(
    *, backend="cpu", model_path=None, rec_batch_num=24, device_id=0, **rapidocr_kwargs
):
    """Full RapidOCR engine; detection/classification on CPU, rec on `backend`.

    Callable as engine(image, use_cls=False, use_det=True, use_rec=True) and also
    exposes .text_det / .text_rec / .backend_info, matching the interface the
    outcome reader used upstream. Pass det_limit_side_len=960 to reproduce the
    upstream outcome OCR geometry.
    """
    from rapidocr_onnxruntime import RapidOCR

    rec = create_ocr_engine(
        backend=backend,
        model_path=model_path,
        rec_batch_num=rec_batch_num,
        device_id=device_id,
    )
    settings = {
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
        "rec_batch_num": rec_batch_num,
    }
    settings.update(rapidocr_kwargs)
    # Provider selection is controlled here only; never let kwargs re-enable
    # RapidOCR's own cuda/dml session paths.
    for stage in ("det", "cls", "rec"):
        settings[f"{stage}_use_cuda"] = False
        settings[f"{stage}_use_dml"] = False
    if model_path is not None:
        settings["rec_model_path"] = str(model_path)
    engine = RapidOCR(**settings)
    engine.text_rec = rec.text_rec
    engine.backend_info = {
        **rec.backend_info,
        "det_backend": "cpu",
        "cls_backend": "cpu",
    }
    return engine
