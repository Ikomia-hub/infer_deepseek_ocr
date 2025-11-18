from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_deepseek_ocr.infer_deepseek_ocr_process import InferDeepseekOcrParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
from torch.cuda import is_available


# --------------------
# - Class which implements widget associated with the algorithm
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferDeepseekOcrWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferDeepseekOcrParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Model name
        self.edit_model = pyqtutils.append_edit(self.grid_layout, "Model name", self.parameters.model_name)

        # Prompt
        self.edit_prompt = pyqtutils.append_edit(self.grid_layout, "Prompt", self.parameters.prompt)

        # CUDA
        self.check_cuda = pyqtutils.append_check(self.grid_layout, "Cuda", self.parameters.cuda and is_available())

        # Mode dropdown replacing base_size/image_size/crop_mode
        self.combo_mode = pyqtutils.append_combo(self.grid_layout, "Mode")
        self.combo_mode.addItem("Tiny")
        self.combo_mode.addItem("Small")
        self.combo_mode.addItem("Base")
        self.combo_mode.addItem("Large")
        self.combo_mode.addItem("Gundam")
        self.combo_mode.setCurrentText(getattr(self.parameters, "mode", "Gundam"))

        # Test compress
        self.check_test_compress = pyqtutils.append_check(self.grid_layout, "Test compress", self.parameters.test_compress)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_apply(self):
        # Apply button clicked slot
        self.parameters.model_name = self.edit_model.text()
        self.parameters.prompt = self.edit_prompt.text()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.mode = self.combo_mode.currentText()
        self.parameters.test_compress = self.check_test_compress.isChecked()
        self.parameters.update = True

        # Send signal to launch the algorithm main function
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build algorithm widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferDeepseekOcrWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the algorithm name attribute -> it must be the same as the one declared in the algorithm factory class
        self.name = "infer_deepseek_ocr"

    def create(self, param):
        # Create widget object
        return InferDeepseekOcrWidget(param, None)
