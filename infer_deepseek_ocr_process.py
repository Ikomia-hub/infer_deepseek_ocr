import copy
import os
from typing import Any

import torch
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

        self.update = (old_model_name != self.model_name) or (old_cuda != self.cuda)

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
        self.output_folder = os.path.join(self.base_dir, "output")
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
        self.device = torch.device("cpu")
        self.img_prompt = "<image>"

    def load_model(self):
        param = self.get_param_object()
        self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")

        # Prefer bfloat16 on CUDA, otherwise float32
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            param.model_name,
            trust_remote_code=True,
            cache_dir=self.model_folder,
        )

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
        image_path = img_input.source_file_path

        # Ensure model is loaded or reload if settings changed
        if self.model is None or self.tokenizer is None or param.update:
            self.load_model()

        with torch.no_grad():
            # Run inference
            # Resolve mode to underlying size/crop configuration
            base_size, image_size, crop_mode = MODES.get(param.mode, MODES["Gundam"])
            res = self.model.infer(
                self.tokenizer,
                prompt=self.img_prompt + param.prompt,
                image_file=image_path,
                output_path=self.output_folder,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                test_compress=param.test_compress,
            )

        # Prepare output
        output_txt = self.get_output(1)
        # res may be a string or dict depending on remote code; normalize to text
        response_text = res if isinstance(res, str) else str(res)
        output_txt.data = {
            "response": response_text,
        }

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
        self.info.version = "1.0.1"
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
