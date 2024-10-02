<!-- Eduardo Lima Barros -->
# <font color='White'>Eduardo Lima Barros</font>
## <font color='White'>Generative IA for LLM - Llama-2-7b-chat-hf</font>
## <font color='White'>Fine-Tuning with QLoRA for sentiment analisys</font>

## Quick disclaimer:
#### Dont use this without a GPU for at least 12 GB, wich you can get for free on Colab, set this configuration:
![image](https://github.com/user-attachments/assets/34df9479-7b48-446d-b5cf-1cfe529ac88a)

## Needed packages and versions


```python
!pip install -q -U watermark
```


```python
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0
```


```python
!pip install -q trl==0.4.7 gradio==3.37.0 protobuf==3.20.3 scipy==1.11.1
```


```python
!pip install -q sentencepiece==0.1.99 tokenizers==0.13.3 datasets==2.16.1
```


```python
%reload_ext watermark
%watermark -a "Fine Tunig Llama2"
```


```python
# Imports
import os
import torch
import datasets
import pandas as pd
from datasets import load_dataset
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
```


```python
# Define the log level for CRITICAL
logging.set_verbosity(logging.CRITICAL)
```


```python
# Shows your GPU Model
if torch.cuda.is_available():
    print('Num of GPUs', torch.cuda.device_count())
    print('GPU Model:', torch.cuda.get_device_name(0))
    print('Total card memory [GB]:',torch.cuda.get_device_properties(0).total_memory / 1e9)
```


```python
# GPU memory Reset
from numba import cuda
device = cuda.get_current_device()
device.reset()
```


```python
# define the dataset
dataset_name = "dataset.csv"
```


```python
# upload dataset
dataset_loaded = load_dataset('csv', data_files = dataset_name, delimiter = ',')
```


```python
# Data in dic format
dataset_loaded
```


```python
# Name of the pre-trained LLM repository
repositorio_hf = "NousResearch/Llama-2-7b-chat-hf"
```


```python
# New model name
model_fine_tuned = "new_model_fine_tuned"
```

## Defining the configuration arguments


```python
# LoRA arguments
lora_r = 32
lora_alpha = 16
lora_dropout = 0.1
```


```python
# bitsandbytes arguments (QLoRa)
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
```


```python
# fine-tuned arguments
output_dir = "output"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
```


```python
# Grouping sequences into batches of the same length
group_by_length = True
save_steps = 0
logging_steps = 400
```


```python
# Accuracy of training data
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
```


```python
# Defining quantization arguments
bnb_config = BitsAndBytesConfig(load_in_4bit = use_4bit,
                                bnb_4bit_quant_type = bnb_4bit_quant_type,
                                bnb_4bit_compute_dtype = compute_dtype,
                                bnb_4bit_use_double_quant = use_nested_quant)
```


```python
# Loading the pre-trained base model
modelo = AutoModelForCausalLM.from_pretrained(repositorio_hf,
                                              quantization_config = bnb_config,
                                              device_map = "auto")
```


```python
# We won't use the cache
modelo.config.use_cache = False
modelo.config.pretraining_tp = 1
```


```python
# Loading the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(repositorio_hf, trust_remote_code = True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```


```python
# Loading the LoRA configuration
peft_config = LoraConfig(lora_alpha = lora_alpha,
                         lora_dropout = lora_dropout,
                         r = lora_r,
                         bias = "none",
                         task_type = "CAUSAL_LM")
```


```python
# Setting training arguments
training_arguments = TrainingArguments(output_dir = output_dir,
                                       num_train_epochs = num_train_epochs,
                                       per_device_train_batch_size = per_device_train_batch_size,
                                       gradient_accumulation_steps = gradient_accumulation_steps,
                                       optim = optim,
                                       save_steps = save_steps,
                                       logging_steps = logging_steps,
                                       learning_rate = learning_rate,
                                       weight_decay = weight_decay,
                                       fp16 = fp16,
                                       bf16 = bf16,
                                       max_grad_norm = max_grad_norm,
                                       max_steps = max_steps,
                                       warmup_ratio = warmup_ratio,
                                       group_by_length = group_by_length,
                                       lr_scheduler_type = lr_scheduler_type)
```


```python
# Defining the arguments of Supervised Fine-Tuning
trainer = SFTTrainer(model = modelo,
                         train_dataset = dataset_loaded['train'],
                         peft_config = peft_config,
                         dataset_text_field = "train",
                         max_seq_length = None,
                         tokenizer = tokenizer,
                         args = training_arguments,
                         packing = False)
```

> Model Training with Fine Tuning


```python
%%time
trainer.train()
```


```python
# Saving the trained model
trainer.model.save_pretrained(model_fine_tuned)
```

<!-- Projeto Desenvolvido na Data Science Academy - www.datascienceacademy.com.br -->


```python
# New input text
prompt = "It's rare that a movie lives up to its hype, even rarer that the hype is transcended by the actual achievement"
```


```python
# Sentiment Analysis Pipeline with Adjusted Model
pipe = pipeline(task = "text-generation",
                model = model,
                tokenizer = tokenizer,
                max_length = 200)
```


```python
# Run the pipeline and extract the result
result = pipe(f"<s>[INST] {prompt} [/INST]")
```


```python
print(result)
```


```python
print(result[0]['generated_text'])
```


```python
# Free your memory
del model
del pipe
del trainer
import gc
gc.collect()
```


```python
# Load the model into fp16 and merge it with the LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(repository_hf,
                                                  low_cpu_mem_usage = True,
                                                  return_dict = True,
                                                  torch_dtype = torch.float16,
                                                  device_map = "auto")
```


```python
# Create the final model
model_fine_tuned_final = PeftModel.from_pretrained(base_model, model_fine_tuned)
```


```python
# Merge and download the model
model_fine_tuned_final = model_fine_tuned_final.merge_and_unload()
```


```python
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(repository_hf, trust_remote_code = True)
tokenizer.pad_token = tokenizer_dsa.eos_token
tokenizer.padding_side = "right"
```


```python
# Template saver and tokenizer
model_fine_tuned_final.save_pretrained('new-model-llm-llama2')
tokenizer.save_pretrained('new-model-llm-llama2')
```


```python
# New input text
prompt = "It's rare that a movie lives up to its hype, even rarer that the hype is transcended by the actual achievement"
```


```python
# Create the pipeline
pipe = pipeline(task = "text-generation",
                model = model_fine_tuned_final,
                tokenizer = tokenizer,
                max_length = 200)
```


```python
# Run the pipeline and extract the result
result = pipe(f"<s>[INST] {prompt} [/INST]")
```


```python
print(result)
```


```python
# Let's not just classify the feeling.
# Let's generate positive and/or negative text from the initial evaluation (text).
print(result[0]['generated_text'])
```


```python
# Frees up GPU memory
from numba import cuda
device = cuda.get_current_device()
device.reset()
```


```python
%watermark -a "Fine Tunig Llama2"
```


```python
#%watermark -v -m
```


```python
#%watermark --iversions
```

# That's all for today folks


```python
from google.colab import drive
drive.mount('/content/drive')
```


```python
!jupyter nbconvert --to markdown Generative_IA_LLM_Llama_two.ipynb
```
