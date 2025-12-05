import copy
import json
import os
import tempfile
from typing import Any

import numpy as np
import torch
from PIL import Image
from ikomia import core, dataprocess, utils
from transformers import AutoModel, AutoTokenizer


# Supported modes mapping -> (base_size, image_size, crop_mode)
MODES = {
    "Tiny": (512, 512, False),
    "Small": (640, 640, False),
    "Base": (1024, 1024, False),
    "Large": (1280, 1280, False),
    "Gundam": (1024, 640, True),
}

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferDeepseekOcrParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "deepseek-ai/DeepSeek-OCR"
        self.cuda = torch.cuda.is_available()
        self.prompt = "<|grounding|>Convert the document to markdown."
        self.mode = "Gundam"
        self.test_compress = True
        self.update = False

    def set_values(self, params):
        old_model_name = getattr(self, "model_name", None)
        old_cuda = getattr(self, "cuda", None)

        self.model_name = str(params["model_name"])
        self.cuda = utils.strtobool(params["cuda"])
        self.prompt = str(params["prompt"])
        # New unified mode parameter; default to Gundam if not provided
        self.mode = str(params.get("mode", "Gundam"))
        self.test_compress = utils.strtobool(params["test_compress"])

        self.update = (old_model_name != self.model_name) or (
            old_cuda != self.cuda)

    def get_values(self):
        params = {
            "model_name": str(self.model_name),
            "cuda": str(self.cuda),
            "prompt": str(self.prompt),
            "mode": str(self.mode),
            "test_compress": str(self.test_compress),
        }
        return params


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferDeepseekOcr(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.add_output(dataprocess.DataDictIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferDeepseekOcrParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model: Any = None
        self.tokenizer: Any = None
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        self.model_folder = os.path.join(self.base_dir, "weights")
        os.makedirs(self.model_folder, exist_ok=True)
        self.device = torch.device("cpu")
        self.img_prompt = "<image>"
        self.temp_image_dir = os.path.join(self.base_dir, "tmp_images")
        os.makedirs(self.temp_image_dir, exist_ok=True)
        self.output_dir = os.path.join(self.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self):
        param = self.get_param_object()
        self.device = torch.device(
            "cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")

        # Prefer bfloat16 on CUDA, otherwise float32
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            param.model_name,
            trust_remote_code=True,
            cache_dir=self.model_folder,
        )

        # Configure tokenizer to avoid warnings
        if self.tokenizer.pad_token is None:
            # Set pad_token to eos_token if not set
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Add a pad token if neither exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model = AutoModel.from_pretrained(
            param.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            use_safetensors=True,
            cache_dir=self.model_folder,
        )

        # Move to device and set eval mode
        self.model = self.model.to(self.device).eval()

        param.update = False

    def init_long_process(self):
        self.load_model()
        super().init_long_process()

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def run(self):
        self.begin_task_run()

        # Parameters
        param = self.get_param_object()

        # Input image
        img_input = self.get_input(0)

        temp_file_path = None
        try:
            # Always convert the input to a temporary PNG file
            image_data = img_input.get_image()
            if image_data is None:
                raise ValueError(
                    "No valid image input found. Please provide either a valid image file path "
                    "or an image array."
                )

            pil_image = None
            if isinstance(image_data, np.ndarray):
                if image_data.size == 0:
                    raise ValueError(
                        "No valid image input found. Please provide either a valid image file path "
                        "or a non-empty image array."
                    )
                if image_data.dtype != np.uint8:
                    if image_data.max() <= 1.0:
                        image_data = (image_data * 255).astype(np.uint8)
                    else:
                        image_data = image_data.astype(np.uint8)

                if image_data.ndim == 2:
                    pil_image = Image.fromarray(image_data)
                elif image_data.ndim == 3:
                    pil_image = Image.fromarray(image_data)
                else:
                    raise ValueError(
                        f"Unsupported image array shape: {image_data.shape}"
                    )
            elif isinstance(image_data, Image.Image):
                pil_image = image_data
            else:
                raise ValueError(
                    f"Unsupported image type: {type(image_data)}"
                )

            os.makedirs(self.temp_image_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                    suffix='.png', delete=False, dir=self.temp_image_dir) as temp_file:
                pil_image.save(temp_file, format='PNG')
                temp_file_path = temp_file.name
                final_image_path = temp_file_path
                print(final_image_path)

            # Ensure model is loaded or reload if settings changed
            if self.model is None or self.tokenizer is None or param.update:
                self.load_model()

            with torch.no_grad():
                # Run inference
                # Resolve mode to underlying size/crop configuration
                base_size, image_size, crop_mode = MODES.get(
                    param.mode, MODES["Gundam"])
                res = self.model.infer(
                    self.tokenizer,
                    prompt=self.img_prompt + param.prompt,
                    image_file=final_image_path,
                    output_path=self.output_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    test_compress=param.test_compress,
                    save_results=True
                )

            # Read the result from output/result.mmd
            result_file = os.path.join(self.output_dir, "result.mmd")
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    response_text = f.read().strip()
            else:
                # Fallback: use res if available, otherwise empty string
                response_text = res if isinstance(res, str) else (
                    str(res) if res is not None else "")

            # Prepare output
            output_txt = self.get_output(1)
            output_txt.data = {
                "response": response_text,
            }

            # Save JSON output
            json_output_path = os.path.join(self.output_dir, "deepseek_output.json")
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump({"response": response_text}, f, indent=2, ensure_ascii=False)

        finally:
            # Clean up temporary file if created
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass  # Ignore cleanup errors

        self.emit_step_progress()
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferDeepseekOcrFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_deepseek_ocr"
        self.info.short_description = "DeepSeek-OCR document OCR to Markdown"
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/VLM"
        self.info.version = "1.1.2"
        self.info.icon_path = "images/icon.png"
        self.info.authors = "DeepSeek-AI"
        self.info.article = "DeepSeek-OCR: Contexts Optical Compression"
        self.info.journal = "arXiv:2510.18234"
        self.info.year = 2025
        self.info.license = "MIT"

        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.15.0"
        # self.info.max_ikomia_version = "0.11.1"

        # Python compatibility
        self.info.min_python_version = "3.9.0"
        # self.info.max_python_version = "3.11.0"

        # URL of documentation
        self.info.documentation_link = "https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"

        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_deepseek_ocr"
        self.info.original_repository = "https://github.com/deepseek-ai/DeepSeek-OCR"

        # Keywords used for search
        self.info.keywords = "OCR,Markdown,DeepSeek,Vision-Language,Document"

        # General type: INFER, TRAIN, DATASET or OTHER
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OCR"

        # Min hardware config
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 16
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create algorithm object
        return InferDeepseekOcr(self.info.name, param)
