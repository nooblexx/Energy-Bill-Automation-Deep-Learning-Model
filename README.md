# Energy-Bill-Understanding -> Introducing BEAM (Billing Extraction & Analysis Model)

Hi, my name is Alex Rios. This is my first machine learning project on Github!

**WORK IN PROGRESS**

### Project Goals

This project explores the development of a model system for understanding and extracting structured information from utility bills using machine learning and document analysis frameworks. The project focuses on:

- Use Optical Character Recognition (OCR) to turn PDFs into structured text preserving layout information for the LLM
- Handle data preprocessing following standard lablel encoding, text cleaning, and data cleaning without the use of a HuggingFace datasets
- Fine-tune a transformer model from a pre-trained LLM to reconginize specific labels
- Model will learn to predict labels of useful information (energy usage/cost)
- Data will be aggregated into JSON format to ready to be read
- Generate excel sheet from output labels from LLM
- Streamlit implementation to make model accessible through a web-application

### Task

The task requires reviewing a electricity bill and extracting key data such as energy usage, demand usage and costs. The process is extremely repetitive and follows recognizable patterns making it ideal for machine learning models to learn from.

### Dataset

Currently, this data is not currently accessible to the public. I will be using data from Lehigh University's Industrial Training and Assessment Center. After some modificatons to the data to make sure it fits guidelines, the dataset will be provided. The model will be trained using utility bills, specifically PPL or PSEG, from manufacturing companies.

### Required Installations

```
!conda create -n ml_beam python=3.11
!conda activate ml_beam
!conda install -c conda-forge pymupdf
!conda install pytorch torchvision
!python -m pip install paddlepaddle==3.2.0
!python -m pip install paddleocr jupyter matplotlib transformers seqeval scikit-learn
```

### Extra Installation used for Training

```
pip install -U label-studio
```
