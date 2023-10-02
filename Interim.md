# Answer Sentence Selection (AS2) - Interim Report

__Team Number:__  18

__Team Name:__  Sesame Street

__Team Members:__   Srija (2021114002), Kapil(2021101028), Harinie(2021114008)

## Description:
As part of the study, we investigated numerous methodologies, and this paper details the datasets we utilised, our approaches, our conclusions, and next steps. We will also go through our committed deadlines as outlined in the project overview.

## Expected Progress as per Timeline:
- Completion of data preprocessing (Completed)
- Begin working on the implementation of the first apprach/paper 

## Progress made:
- We first approach the understanding of the architecture presented in the paper: __A Study on Efficiency, Accuracy and Document Structure
for Answer Sentence Selection__
- We have performed exploratory data anlaysis to identify any patterns and applied tokenization by ensuring we filter noise in the data and finally we applied embeddings to convert text to meaningful representation (especially for the WikiQA dataset)
- We have also implemented the word relatedness part of the Cosinet architecture as described in the pair, to allow us to apply static attention based on similarity between the question and the answer sentence words.
  

## Exploratory Data Analysis(EDA):
Dataset downloaded from hugging face repo originally extracted from STS Benchmark and the data available at these location:   
https://huggingface.co/datasets/wiki_qa  
https://www.microsoft.com/en-us/download/details.aspx?id=52419

We also worked with the SQuAD dataset and performed preprocessing for the same. 

The datasets (after preprocessing) have been uploaded to drive and links to them have been included in the respective ipynb notebooks.


## Tokenization:
As part of tokenization, the following preprocessing processes were used:

- Remove punctuation
- Replace numbers with num tag
- Use Lower case
- Used Applied Lemmatization
- Used Applied Stemming
- Unknown treatment by replacing all words with less frequency with the unk tag
- The length of the sentences was considerably decreased after using the preceding preparation processes
- Removal of questions with no answers

## Embeddings:
As per the paper 

## Next Steps:

### 1. Implementation of Approach 2: TANDA
- In a nutshell, TANDA is a technique for fine-tuning pre-trained Transformer models sequentially in two steps:

  - first, transfer a pre-trained model to a model for a general task by fine-tuning it on a large and high-quality dataset.
  - then, perform a second fine-tuning step to adapt the transferred model to the target domain.  

- ASNQ is a dataset for answer sentence selection derived from Google Natural Questions (NQ) dataset (Kwiatkowski et al. 2019). We plan to use it to transfer the pretrained models mentioned in the above paper.

### 2. Other Base-Line Models
- For baseline models, 

### 3. Comparison and Analysis