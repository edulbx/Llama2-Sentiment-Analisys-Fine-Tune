### 1. Project for making an Sentiment Analisys



```python
# @title ####1.1 Installs {"form-width":" 1px"}
# some conflicts are soloved just by letting pip deal with some dependencies.
# The way to do that is calling pip just one time with all the packs needed.
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 gradio==3.37.0 protobuf==3.20.3 scipy==1.11.1 sentencepiece==0.1.99 tokenizers==0.13.3 datasets==2.16.1
```


```python
# @title #### 1.2 Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          HfArgumentParser,
                          TrainingArguments,
                          pipeline,
                          logging)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import warnings
warnings.filterwarnings('ignore')
import torch
import os
import torch
import datasets
from datasets import load_dataset
```

### 2. We start by making some configurations



```python
# @title ####2.1 logging level to Critical and verifying GPU
# defining logging level to Critical
logging.set_verbosity(logging.CRITICAL)
# verifing GPU
if torch.cuda.is_available(): #this func pratically writes itself on colab, is just for verifying the GPU
    print('Numbers of GPUs:', torch.cuda.device_count())
    print('GPU Model:', torch.cuda.get_device_name(0))
    print('Total Memory [GB] of GPU:',torch.cuda.get_device_properties(0).total_memory / 1e9)
```

    Numbers of GPUs: 1
    GPU Model: Tesla T4
    Total Memory [GB] of GPU: 15.835660288



```python
# @title ####2.2 Reset of Vram - GPU
#Reset of Vram - GPU - here's a link for you know more about ram and Vram - https://www.techtarget.com/searchstorage/definition/video-RAM#:~:text=In%20simplest%20terms%2C%20VRAM%20is,to%20processing%20graphics%2Drelated%20tasks.
#from numba import cuda
#device = cuda.get_current_device()
#device.reset()
```

###3. Now we load the data


```python
# Define o nome do dataset
dataset = "dataset.csv"
# Load the dataset with the fun load_dataset, arg are type of file, name of dataset, and delimiter.
dataset_loaded = load_dataset('csv', data_files = dataset, delimiter = ',')

dataset_loaded
# This function will deliver the dataset and dic format, take a look:
## DatasetDict({                        a tuple ()
 ##    train: Dataset({                 inside is a dictnary {}
  ##       features: ['train'],         inside we have a list
   ##      num_rows: 17057
    ##})
  ##})

# it's just an approximation for you to realize what's happening with python structures.
```


    Generating train split: 0 examples [00:00, ? examples/s]





    DatasetDict({
        train: Dataset({
            features: ['train'],
            num_rows: 17057
        })
    })



#### 4. Now we import the model


```python

#we get it from the hf repo

hf_repo = "NousResearch/Llama-2-7b-chat-hf"

#defining the name for the new model (after fine tuning)
model = "model_fine_tuned"
```

#### 5. Now we set the fine tune arguments


```python
#Lora:

lora_r = 32
lora_alpha = 16
lora_dropout = 0.1

#BitsAndBytes(Qlora) - quantization:
#by using the quantization method
#with the lora arguments we are making
#the Qlora technique

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4" # specific type created by BitsandBytes
use_nested_quant = False

#fine tune configuration for training:
output_directory = "output"
train_epochs = 1
fp16 = True #if set to true the bf16 must be false
bf16 = False
per_device_train_batch_size = 4
per_device_evaluation_batch_size = 4
gradient_acumulation_steps = 1
gradient_checkpoint = True
max_grad_norm = 0.3
lr = 2e-4
weight_decay = 0.001
optimizer = "Paged_Adamw_32bit"
lr_schedule_type = "cosine"
max_steps = 1
warmup_ratio = 0.03

#grouping sequences by length:
group_by_length = True
save_steps = 0
logging_steps = 400

#data type for computation during the training using the Pytorch library
#you remember that we defined the bnb_4bit_compute_dtype above as float16
#and the torch is a ref to the Pytorch library:
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
```

#### 6. Now we initialize the body of the code with the functions receiving our previously defined arguments.




```python
#BitsAndBytes func
bnb_config = BitsAndBytesConfig(
    load_in_4bit = use_4bit,
    bnb_4bit_quant_type = bnb_4bit_quant_type,
    bnb_4bit_compute_dtype = compute_dtype,
    bnb_4bit_use_double_quant = use_nested_quant)

#loading the model and using the func above
model = AutoModelForCausalLM.from_pretrained(
    hf_repo,
    quantization_config = bnb_config,
    device_map = "auto")

#definig the we wont use cache
model.config.use_cache = False
model.config.pretrainig_tp = 1

#loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#loading the peft config
peft_config = LoraConfig(
    lora_alpha = lora_alpha,
    lora_dropout = lora_dropout,
    r = lora_r,
    bias = "none",
    task_type = "CAUSAL_LM")

#the function for the trainig args defined above:
training_args = TrainingArguments(
    output_dir = output_directory,
    num_train_epochs = train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    gradient_accumulation_steps = gradient_acumulation_steps,
    optim = optimizer,
    save_steps = save_steps,
    logging_steps = logging_steps,
    learning_rate = lr,
    weight_decay = weight_decay,
    fp16 = fp16,
    bf16 = bf16
    max_grad_norm = max_grad_norm,
    max_steps = max_steps,
    warmup_ratio = warmup_ratio,
    group_by_length = group_by_length,
    lr_scheduler_type = lr_schedule_type)  #Yeah I know it's many arguments but hey are need don't get confuse,
    #use a good text editor for coding or some good IDE and you be able to review more easly those arguments above.

# now we define the arguments for the supervised fine-tunig:

trainer = SFTTrainer( #we imported this package for this purpose
    model = model,
    train_dataset = dataset_loaded["train"], #defined in nº 3
    peft_config = peft_config, #function above
    dataset_info_text_field = "train" #used to identify the column in the dataset that contains the text for the model to learn from
    max_seq_length = None,
    tokenizer = tokenizer,
    args = training_args
    packing = False)
```

#### 7. Now to finally begin the trainig and save the model


```python
#"%%time" measure the execution time
%%time
trainer.train() #this will call our function and the method .train to begin the training
```


```python
trainer.model.save_pretrained(model)
```

#### 8. Testing for deploy and creating a pipeline


```python
#new prompt to be used
prompt = "It's rare that a movie lives up to its hype, even rarer that the hype is transcended by the actual achievement"
```


```python
#pipeline for the Sentiment Analisys with the fine-tuned model
pipe = pipeline(task = "text-generation", model = model, tokenizer = tokenizer, max_length = 200) #you can adjust the max_length

#result
result  = pipe(f"<s>[INST] {prompt} [INST] ")

print(result)
print(result[0]['generated_text'])


```


```python
#to clear the mem
del model
del pipe
del trainer
import gc
gc.collect()
```

#### 9. Merging the base model with the new LoRA weigth


```python
#first we load the base model with fp16 and merge with LORA
base_model = AutoModelForCausalLM.from_pretrained(
    hf_repo,
    low_cpu_mem_usage = True,
    return_dict = True,
    torch_dtype = torch.float16,
    device_map = "auto",
)

#now we creat the final model
final_model = PeftModel.from_pretrained(base_model, model) #we passing the base model and our model
#now we merge and unload
final_model = final_model.merge_and_unload() #we are merging the models above

#we will need the tokenizer once again
tokenizer = AutoTokenizer.from_pretrained(hf_repo, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#we save the model
final_model.save_pretrained("new_model")
tokenizer.save_pretrained("new_model")

```


```python
#we teste again:
prompt = "It's rare that a movie lives up to its hype, even rarer that the hype is transcended by the actual achievement"
pipe = pipeline(task = "text-generation",
                model = final_model,
                tokenizer = tokenizer,
                max_length = 200)
result = pipe(f"<s>[INST] {prompt} [INST] ")
print(result)
print(result[0]['generated_text'])
```


```python
#just for cleaning the GPU mem
from numba import cuda
device = cuda.get_current_device()
device.reset()
```

**### And we are done, thank you**


```python
!jupyter nbconvert --to markdown Sentiment_Analisys.ipynb
```

    [NbConvertApp] WARNING | pattern 'Sentiment_Analisys.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

