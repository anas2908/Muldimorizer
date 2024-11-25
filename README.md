# Muldimorizer


## Folder Structure

### 1. **`ablation/`**
This folder contains code implementing traditional methods of multimodal integration. Specifically, it integrates text, video, and audio features by adding before passing them to the language model encoder. This serves as a baseline to compare against our novel triangulated attention mechanism.

---

### 2. **`baseline/`**
Includes code for the **LLaVA** baseline, which processes videos by extracting and using the middle frame instead of handling entire video sequences. The baselines demonstrate the effectiveness of simplified approaches and offer performance comparisons.

---

### 3. **`model/`**
This folder contains two critical Python files for the core modeling:
- **`modeling_shemud_bart.py`:** Backend implementation based on BART, incorporating triangulated attention in the encoder layer for seamless integration of audio and visual embeddings.
- **`rouge.py`:** Tracks evaluation metrics during training, including ROUGE-L, to measure the quality of generated summaries.

---

### 4. **`preprocessing_codes/`** & **`preprocessing_excel/`**
Handles all preprocessing tasks for datasets, including text, video, and audio processing:
- Extracts video embeddings using **OpenAI CLIP**.
- Computes audio features using **MFCCs**.
- Prepares the dataset for input to the model.
- Preprocessed Dataset also provided within the folder

---

### 5. **Root Files**
- **`gradio_temp.py`:** Inference code with a user-friendly UI for interactive testing using Gradio.
- **`infer_alltemp.py`:** Inference script without UI for batch processing.
- **`SCORE_CALCULATION.py`:** Code for calculating evaluation metrics, including:
  - BLEU
  - ROUGE-L
  - CIDEr
  - METEOR
  - BERTScore
  - And more...

- **`muldimorizer.py`:** The **frontend** of the model that processes preprocessed data and loads video/audio embeddings into the BART backend. If specific modalities (e.g., video or audio) are not required, simply exclude their input paths in this script, and the system will adapt automatically.

---

## Key Features

- **Triangulated Attention:** 
  Integrates audio and visual embeddings directly in the encoder layer for improved multimodal interaction.

- **Dynamic Modality Selection:** 
  The model automatically adapts to the absence of specific modalities by handling missing inputs seamlessly at the BART frontend.

- **Evaluation Metrics:** 
  Comprehensive metric suite for analyzing performance, including BLEU, ROUGE-L, CIDEr, METEOR, and BERTScore.

- **Pretrained Features:**
  - **Video:** Processed with OpenAI CLIP.
  - **Audio:** Extracted using simple MFCCs.

---

## Acknowledgements

- **[M2H2 Paper](https://github.com/declare-lab/M2H2-dataset):** Provided the dataset for training and evaluation.
- **[MeSum Paper (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.389/):** Inspired the architectural framework for this project.

---

## How to Use

1. **Preprocessing:** 
   Use the scripts in `preprocessing_codes/` and `preprocessing_excel/` to prepare the dataset.

2. **Model Training:** 
   Modify and run files in the `model/` folder to train the backend with triangulated attention.

3. **Inference:**
   - Interactive: Run `gradio_temp.py` for a Gradio-based interface.
   - Batch: Run `infer_alltemp.py`.

4. **Evaluation:** 
   Analyze results using `SCORE_CALCULATION.py` to compute multiple metrics.

For further customization, dynamically enable/disable modalities at the BART frontend by including or excluding video/audio paths.
