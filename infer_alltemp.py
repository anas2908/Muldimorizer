raw_text = "yeah"
raw_speaker = "Dilruba"
raw_label = "non-humour"
raw_audio_path = "alltemphere/inferfiles/4b.wav"   # audio_feature_preprocessed
raw_video_path = "alltemphere/inferfiles/4b.mp4" 

import os
import librosa
import numpy as np

# Function to extract MFCC embeddings
def extract_mfcc(file_path, n_mfcc=13):
    # Load the .wav file
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # Return the MFCC features as a numpy array
    return mfccs

# Input path for the single audio file
# raw_audio_path = r"path\to\your\audio_file.wav"  # replace with the actual path to the audio file

# Check if the .wav file exists
if os.path.exists(raw_audio_path):
    # Extract MFCC embeddings and store in a variable
    audio_feature = extract_mfcc(raw_audio_path)
    print("MFCC embeddings extracted and stored in variable 'audio_feature'.")

    # Flatten the array and then reshape it to (1, 1024)
    flattened_data = audio_feature.flatten()[:1024]  # Truncate if more than 1024 elements
    audio_feature_preprocessed = np.pad(flattened_data, (0, max(0, 1024 - flattened_data.size)), 'constant')  # Pad if less than 1024
    audio_feature_preprocessed = audio_feature_preprocessed.reshape(1, 1024)

    print(f"The new shape of the array is: {audio_feature_preprocessed.shape}")
else:
    print(f"File not found: {raw_audio_path}")

print("File processing completed.")


import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract frames from a video at 1 fps
def extract_frames(video_path, fps=1):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_list = []
    count = 0
    while success:
        if count % int(vidcap.get(cv2.CAP_PROP_FPS)) == 0:  # Capture one frame per second
            frame_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert to RGB for CLIP
        success, image = vidcap.read()
        count += 1
    vidcap.release()
    return frame_list

# Function to extract CLIP embeddings from frames
def extract_clip_embeddings(frames):
    embeddings_list = []
    for frame in frames:
        pil_image = Image.fromarray(frame)
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        embeddings_list.append(embeddings.cpu().numpy().flatten())  # Flatten to [1, 512]
    return np.array(embeddings_list)  # Shape [n_frames, 512]

# Path for the single video file
# raw_video_path = r"path\to\your\video_file.mp4"  # replace with the actual path to the video file

# Check if the .mp4 file exists
if os.path.exists(raw_video_path):
    # Step 1: Extract frames from the video at 1 fps
    frames = extract_frames(raw_video_path, fps=1)

    # Step 2: Extract CLIP embeddings from the frames and store them in 'video_feature'
    video_feature = extract_clip_embeddings(frames)
    print(video_feature)
    # Step 3: Flatten, pad/truncate, and reshape to (1, 1024)
    flattened_data = video_feature.flatten()[:1024]  # Truncate if more than 1024 elements
    video_feature_preprocessed = np.pad(flattened_data, (0, max(0, 1024 - flattened_data.size)), 'constant')  # Pad if less than 1024
    video_feature_preprocessed = video_feature_preprocessed.reshape(1, 1024)

    print(f"The new shape of the array is: {video_feature_preprocessed.shape}")
else:
    print(f"File not found: {raw_video_path}")

print("File processing completed.")


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
# from modeling_meme_bart_anas_third import BartForConditionalGeneration

from modeling_shemud_bart import BartForConditionalGeneration

metric = load_metric("alltemphere/rouge.py")

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")


print("\nDEVICE:\t",device)


dataset_full=[]

raw_text = [raw_text]
raw_speaker = [raw_speaker]
raw_label = [raw_label]


dataset_full.append({"raw_text": str(raw_text[0]),"raw_speaker": str(raw_speaker[0]), "raw_label":str(raw_label[0])}) #palak "trimming_sec" should be "answers"


def transform_single_dialogsumm_file(file):
    result = {"raw_text":[], "raw_speaker":[], "raw_label":[]}  #palak replace the strings with the column names

    for i in range(len(file)):
        result["raw_text"].append(file[i]["raw_text"])
        result["raw_speaker"].append(file[i]["raw_speaker"])
        result["raw_label"].append(file[i]["raw_label"])
    return Dataset.from_dict(result)

def transform_test_file(file):
    result = {"raw_text":[], "raw_speaker":[], "raw_label":[]}  #palak replace the strings with the column names

    for i in range(len(file)):
        result["raw_text"].append(file[i]["raw_text"])
        result["raw_speaker"].append(file[i]["raw_speaker"])
        result["raw_label"].append(file[i]["raw_label"])
    return Dataset.from_dict(result)

def transform_dialogsumm_to_huggingface_dataset(test):
    test = transform_test_file(test)
    return DatasetDict({"test":test})




model_checkpoint = 'alltemphere/BART_42_LARGE_mfccs_clip_inTok421024_3e-5_batchS_32ROG_ep_11/checkpoint-1738'



# metric = load_metric("rouge.py")

TEST_SUMMARY_ID = 1



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


max_input_length = 1024


filename_model = "muldimorizer_inference"

print(filename_model)

# path = "/content/drive/MyDrive/final_project_fml_nlp/Triangulated Attention/"
# MODEL_PATH = path+"Model Path/"+filename_model

is_cuda = torch.cuda.is_available()



import pickle


# filename_dataset="memes.xlsx"

model = BartForConditionalGeneration.from_pretrained(model_checkpoint)  #palak comment this
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


max_target_length = 512 * 2

# path = "/home/arpan_2121cs33/anas/memes"


from sklearn.model_selection import train_test_split
import random


# train_size = 0.0
# val_size = 0.0
# test_size = 1.0

# train_data, val_test_data = train_test_split(dataset_full, train_size=train_size, random_state=42)
# val_data, test_data = train_test_split(val_test_data, train_size=val_size/(val_size + test_size), random_state=42)
# raw_datasets = transform_dialogsumm_to_huggingface_dataset(test_data)

# Use the entire dataset as test_data without any splitting
test_data = dataset_full.copy()  # Copy the entire dataset into test_data

# Now use test_data to create the Hugging Face dataset
raw_datasets = transform_dialogsumm_to_huggingface_dataset(test_data)






def preprocess_function(examples):
    inputs = []
    compressed_vid = []
    compressed_aud = []
    model_inputs = {}

    for (raw_text) in zip(examples["raw_text"], examples["raw_speaker"]):
        # inputs.append(str("bot is provided with English Utterance: " + str(raw_text) + "bot is also provided with the speaker : " + str(raw_speaker) +
        #                   " bot is also provided with video and audio embeddings, bot task is to classify either it is 'humour' or 'non-humour' and write a summary"))
        inputs.append(str("bot is provided with English Utterance: " + str(raw_text)  + "bot is also provided with the speaker : " + str(raw_speaker) + ", the scene is " + str(raw_label) +
                          " bot is also provided with video and audio embeddings, bot task is to write a summary"))
        try:
            # Load the .npy file instead of .pt
            # npy_fileload = np.load('/content/drive/MyDrive/final_project_fml_nlp/Triangulated Attention/shemud/M2H2-dataset-main/Main-Dataset/VideoPreprocessedReshaped/' + str(uid) + ".npy")
            # print("Shape of video loaded tensor:", npy_fileload.shape)
            compressed_vid.append(video_feature_preprocessed)

        except Exception as e:
            print(e)
            compressed_vid.append(np.zeros((1, 1024), dtype=np.float32))  # Ensure the type matches


        try:
              # Load the .npy file instead of .pt
              # npy_fileload = np.load('/content/drive/MyDrive/final_project_fml_nlp/Triangulated Attention/shemud/M2H2-dataset-main/Main-Dataset/AudioPreprocessedReshaped/' + str(uid) + ".npy")
              # print("Shape of Audio loaded tensor:", npy_fileload.shape)

              # if npy_fileload.shape != (1, 1024):  # Adjust shape check for numpy array
                  # compressed_aud.append(np.zeros((1, 1024), dtype=np.float32))
              # else:
              compressed_aud.append(audio_feature_preprocessed)

        except Exception as e:
            print(e)
            compressed_aud.append(np.zeros((1, 1024), dtype=np.float32))  # Ensure the type matches

    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    model_inputs["video_embedd"] = compressed_vid
    model_inputs["audio_embedd"] = compressed_aud

    label_input = []
    for cap in examples["raw_text"]:
        label_input.append(cap)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(label_input, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
print(raw_datasets)
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

batch_size = 1
args = Seq2SeqTrainingArguments(
    "ROG_",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,
    predict_with_generate=True,
    save_strategy="epoch",
    metric_for_best_model="eval_rouge1",
    greater_is_better=True,
    seed=42,
    generation_max_length=max_target_length,
    logging_strategy = "epoch",report_to="wandb"
)
trainer = Seq2SeqTrainer(
    model,
    args,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)


import nltk
import numpy as np



# out = trainer.predict(tokenized_datasets["test"],num_beams=5).to(device)
out = trainer.predict(tokenized_datasets["test"],num_beams=5)

predictions, labels ,metric= out
print(metric)


decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

#### decoded_preds[0]######   is our ultimate summary


print(decoded_preds[0])
