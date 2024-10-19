<!-- Eduardo Lima Barros -->
# <font color='White'>Eduardo Lima Barros</font>
## <font color='White'>Generative IA for LLM - Llama-2-7b-chat-hf</font>
## <font color='White'>Fine-Tuning with QLoRA for sentiment analisys</font>

![InShot_20240820_224138456](https://github.com/edulbx/Llama2-Sentiment-Analisys-Fine-Tune/blob/main/fine%20tuned%20gif.gif)

## Quick disclaimer:
#### Dont use this without a GPU for at least 12 GB, wich you can get for free on Colab, set this configuration:
![image](https://github.com/user-attachments/assets/34df9479-7b48-446d-b5cf-1cfe529ac88a)


### 1. Project for making an Sentiment Analisys


#### 1.1 Installs 
```python

# some conflicts are soloved just by letting pip deal with some dependencies.
# The way to do that is calling pip just one time with all the packs needed.
!pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 gradio==3.37.0 protobuf==3.20.3 scipy==1.11.1 sentencepiece==0.1.99 tokenizers==0.13.3 datasets==2.16.1
```

#### 1.2 Imports
```python

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


####  2.1 logging level to Critical and verifying GPU
```python

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


#### 2.2 Reset of Vram - GPU
```python

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
