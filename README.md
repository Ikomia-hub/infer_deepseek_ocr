<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_deepseek_ocr</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_deepseek_ocr">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_deepseek_ocr">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_deepseek_ocr/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_deepseek_ocr.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) by DeepSeek AI, use groundbreaking approach to compressing long contexts via optical 2D mapping. This innovative system demonstrates that vision-based compression can achieve remarkable efficiency in handling text-heavy documents, potentially revolutionizing how large language models (LLMs) process extensive textual information.

The DeepSeek-OCR system consists of two primary components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Together, they achieve an impressive 97% OCR precision when compressing text at a ratio of less than 10× (meaning 10 text tokens compressed into 1 vision token).

![benchmark](https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/assets/fig1.png?raw=true)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deepseek_ocr", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/NanoNets/Nanonets-OCR2/blob/main/assets/bank_statement.jpg?raw=true")

# Display input
display(algo.get_input(0).get_image())
# Save output .json
qwen_output = algo.get_output(1)
qwen_output.save('deepseek_output.json')
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (string, default: `deepseek-ai/DeepSeek-OCR`): Hugging Face model repo to load.
- **cuda** (bool, default: auto): Use GPU if available. If set to True but no CUDA device is present, it will fall back to CPU.
- **prompt** (string, default: `"<|grounding|>Convert the document to markdown."`): Text instruction appended after the image token to control the output style and task.
- **mode** (enum, default: `Gundam`): Preset controlling resolution and cropping. One of: `Tiny`, `Small`, `Base`, `Large`, `Gundam`.

        - Gundam (Recommended): Balanced performance with crop mode # base_size = 1024, image_size = 640, crop_mode = True
        - Base: Standard quality without cropping                   # base_size = 1024, image_size = 1024, crop_mode = False
        - Large: Highest quality for complex documents              # base_size = 1280, image_size = 1280, crop_mode = False
        - Small: Faster processing, good for simple text            # base_size = 640, image_size = 640, crop_mode = False
        - Tiny: Fastest, suitable for clear printed text            # base_size = 512, image_size = 512, crop_mode = False  
- **test_compress** (bool, default: `True`): Enable internal compression/fast path to reduce compute and VRAM. Turn off for maximum fidelity.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deepseek_ocr", auto_connect=True)

algo.set_parameters({
    'prompt': "<|grounding|>Convert the document to markdown.",
    'mode': "Gundam",
    'test_compress': "True",
    })

# Run on your image  
wf.run_on(url="https://github.com/NanoNets/Nanonets-OCR2/blob/main/assets/bank_statement.jpg?raw=true")

# Show input
display(algo.get_input(0).get_image())
# Save output .json
qwen_output = algo.get_output(1)
qwen_output.save('deepseek_output.json')
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_deepseek_ocr", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/NanoNets/Nanonets-OCR2/blob/main/assets/bank_statement.jpg?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Advanced usage 

### 💡 Tips for Best Results
- **For receipts**: Use "ocr" mode with "gundam" or "base" preset
- **For documents with tables**: Use "markdown" mode with "large" preset
- **If text is not detected**: Try different presets in this order: gundam → base → large
- **For handwritten text**: Use "large" preset for better accuracy
- Ensure images are clear and well-lit for optimal results


### :pencil: Prompts examples

- **document**: <|grounding|>Convert the document to markdown.
- **other image**: <|grounding|>OCR this image.
- **without layouts**: Free OCR.
- **figures in document**: Parse the figure.
- **general**: Describe this image in detail.
- **rec**: Locate <|ref|>xxxx<|/ref|> in the image.
- '先天下之忧而忧'
