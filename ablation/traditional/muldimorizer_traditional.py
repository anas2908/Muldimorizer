import json
from datasets import Dataset,DatasetDict, load_metric
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer#, BartForConditionalGeneration
import os
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer,  GPTNeoForCausalLM, GPT2LMHeadModel,T5Tokenizer, T5Model,T5ForConditionalGeneration
from copy import deepcopy
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import json
from transformers import DataCollator,DefaultDataCollator

# import sys
# sys.path.append('/raid/nlp/tousin/KM-BART-ACL/alltemphere/model/')

from modeling_traditional import BartForConditionalGeneration

metric = load_metric("rouge.py")

def transform_single_dialogsumm_file(file):
    result = {"uid":[],"hindi":[],"english":[],"goldlabel":[], "speaker":[], "summary":[]}  #palak replace the strings with the column names
    for i in range(len(file)):

        result["uid"].append(file[i]["uid"])
        result["hindi"].append(file[i]["hindi"])
        result["english"].append(file[i]["english"])
        result["goldlabel"].append(file[i]["goldlabel"])
        result["speaker"].append(file[i]["speaker"])
        result["summary"].append(file[i]["summary"])

    return Dataset.from_dict(result)

def transform_test_file(file):
    result = {"uid":[],"hindi":[],"english":[],"goldlabel":[], "speaker":[], "summary":[]}  #palak replace the strings with the column names
    for i in range(len(file)):

        result["uid"].append(file[i]["uid"])
        result["hindi"].append(file[i]["hindi"])
        result["english"].append(file[i]["english"])
        result["goldlabel"].append(file[i]["goldlabel"])
        result["speaker"].append(file[i]["speaker"])
        result["summary"].append(file[i]["summary"])

    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(train,validation,test):
    train = transform_single_dialogsumm_file(train)
    validation = transform_single_dialogsumm_file(validation)
    test = transform_test_file(test)
    return DatasetDict({"train":train,"validation":validation,"test":test})


max_input_length = 1024
num_epochs = 11
batch_size = 32 #12

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")


print("\nDEVICE:\t",device)


model_checkpoint = 'facebook/bart-large'

# path = "/raid/nlp/tousin/KM-BART-ACL/alltemphere/"
bert_model = "BART"
config = "mfccs_clip_inTok42"+str(max_input_length)+"_3e-6_batchS_"+str(batch_size)
filename_model= bert_model+"_ep_"+str(num_epochs)+config
print(filename_model)
MODEL_PATH_CHECKPOINT = "Model Path/"+filename_model+"_Loss_Checkpoints.pt"
MODEL_PATH = "Model Path/"+filename_model

import pickle
filename_dataset="model/MasterExcelFinal.xlsx"
data = pd.read_excel(filename_dataset)
dataset_full=[]

uid = data["uid"].values
hindi = data["hindi"].values
english = data["english"].values
goldlabel = data["goldlabel"].values
speaker = data["speaker"].values
summary = data["summary"].values


for i in range(len(uid)):
    dataset_full.append({"uid": str(uid[i]),"hindi": str(hindi[i]),
                        "english": str(english[i]),"goldlabel": str(goldlabel[i]),"speaker": str(speaker[i]),"summary": str(summary[i]) })



from sklearn.model_selection import train_test_split
import random
train_size = 0.8
val_size = 0.1
test_size = 0.1

train_data, val_test_data = train_test_split(dataset_full, train_size=train_size, random_state=42)
val_data, test_data = train_test_split(val_test_data, train_size=val_size/(val_size + test_size), random_state=42)
print(f"Train size: {len(train_data)}")
print(f"Validation size: {len(val_data)}")
print(f"Test size: {len(test_data)}")

raw_datasets = transform_dialogsumm_to_huggingface_dataset(train_data,val_data,test_data)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)


for param in model.parameters():
    param.requires_grad = True


import numpy as np

max_target_length = 1024  # 512 * 2

def preprocess_function(examples):
    inputs = []
    compressed_vid = []
    compressed_aud = []
    model_inputs = {}

    for (uid, hindi, english, speaker,summary,goldlabel) in zip(examples["uid"], examples["hindi"], examples["english"], examples["speaker"], examples["summary"], examples["goldlabel"]):
        print(f"Processing UID: {uid}")
        # inputs.append(str("bot is provided with English Utterance: " + str(english) + " bot is also provided with Hindi Utterance: " + str(hindi) +
        #                   " bot is also provided with video and audio embeddings, bot task is to classify either it is 'humour' or 'non-humour' thats all"))
        # inputs.append(str("bot is provided with English Utterance: " + str(english)  + "bot is also provided with the speaker : " + str(speaker) +
        #                   " bot is also provided with video and audio embeddings, bot task is to classify either it is 'humour' or 'non-humour' and write a summary"))
        inputs.append(str("bot is provided with English Utterance: " + str(english)  + "bot is also provided with the speaker : " + str(speaker) + ", the scene is " + str(goldlabel) +
                          " bot is also provided with video and audio embeddings, bot task is to write a summary"))
        # inputs.append(str("bot is provided with English Sentence : " + str(english) + " bot is also provided with Chest X-Ray visual embeddings, bot task is to generate detail Report including Findings and Impression"))

        try:
            # Load the .npy file instead of .pt
            npy_fileload = np.load('VideoPreprocessedReshaped/VideoPreprocessedReshaped/' + str(uid) + ".npy")
            # print("Shape of video loaded tensor:", npy_fileload.shape)
            compressed_vid.append(npy_fileload)

        except Exception as e:
            print(e)
            compressed_vid.append(np.zeros((1, 512), dtype=np.float32))  # Ensure the type matches


        try:
              # Load the .npy file instead of .pt
              npy_fileload = np.load('AudioPreprocessedReshaped/AudioPreprocessedReshaped/' + str(uid) + ".npy")
              # print("Shape of Audio loaded tensor:", npy_fileload.shape)

              if npy_fileload.shape != (1, 1024):  # Adjust shape check for numpy array
                  compressed_aud.append(np.zeros((1, 1024), dtype=np.float32))
              else:
                  compressed_aud.append(npy_fileload)

        except Exception as e:
            print(e)
            compressed_aud.append(np.zeros((1, 1024), dtype=np.float32))  # Ensure the type matches

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    model_inputs["video_embedd"] = compressed_vid
    model_inputs["audio_embedd"] = compressed_aud

    label_input = []
    for cap in examples["summary"]:
        label_input.append(cap)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(label_input, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Apply the preprocessing to the datasets
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


args = Seq2SeqTrainingArguments(
    "BART_42_LARGE_"+config+"ROG_ep_"+str(num_epochs),# "BART_LARGE_"+config+"ROG_ep_"+str(num_epochs),
    evaluation_strategy = "epoch",
    eval_steps = 50, # Evaluation and Save happens every 50 steps
    # load_best_model_at_end=True,
    learning_rate=3e-5, #3e-5
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    save_strategy="epoch",
    # metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    fp16=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch"
    # ,report_to="wandb"
)


def collate_fn(batch):
    inputs = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    mask = [item['attention_mask'] for item in batch]
    video_embeds = [item['video_embedd'] for item in batch]
    audio_embeds = [item['audio_embedd'] for item in batch]


    inputs_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in inputs], batch_first=True, padding_value=0)

    labels_text = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in labels], batch_first=True, padding_value=0)

    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in mask], batch_first=True, padding_value=0)

    video_embeds_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in video_embeds], batch_first=True, padding_value=0)

    audio_embeds_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(lst) for lst in audio_embeds], batch_first=True, padding_value=0)

    return {'input_ids': inputs_text, 'labels': labels_text, 'video_embedd': video_embeds_padded, 'audio_embedd': audio_embeds_padded, 'attention_mask': attention_mask}

#data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
from transformers import DataCollator,DefaultDataCollator
data_collator = DefaultDataCollator()



import nltk
# nltk.download('punkt_tab')
nltk.download('punkt')



import nltk
import numpy as np
import re
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v , 4) for k, v in result.items()}




trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset= tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    #data_collator=data_collator,
    data_collator=collate_fn,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
    # compute_metrics=compute_metrics_F1overlap
    compute_metrics=compute_metrics
)




trainer.train()




out = trainer.predict(tokenized_datasets["test"],num_beams=5)

predictions, labels ,metric= out
print(metric)


decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after e ach sentence
decoded_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
decoded_labels = [" ".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]




with open("test_output.txt","w") as f:
    for i in decoded_preds:
        print(i)
        f.write(str(i.replace("\n",""))+"\n")

    # """
with open(MODEL_PATH+"_report.json", "a") as outfile:
    outfile.write('[')
    for index, item in enumerate(decoded_preds):
        dictionary = {
            "Gold_report": str(decoded_labels[index]),
            "Generated_report": decoded_preds[index]
        }
        print(dictionary)
        if index > 0:
            outfile.write(',')
        json.dump(dictionary, outfile)
    outfile.write(']')



