# Answer Sentence Selection (AS2) - Interim Report

__Team Number:__  18

__Team Name:__  Sesame Street

__Team Members:__   Srija (2021114002), Kapil(2021101028), Harinie(2021114008)

## Description:
As part of the study, we investigated numerous methodologies, and this paper details the datasets we utilised, our approaches, our conclusions, and next steps. We will also go through our committed deadlines as outlined in the project overview.

## Expected Progress as per Timeline:
- Completion of data preprocessing (Completed)
- Begin working on the implementation of the first approach/paper (Completed)

## Progress made:
- We first approach the understanding of the architecture presented in the paper: __A Study on Efficiency, Accuracy and Document Structure
for Answer Sentence Selection__
- We have performed exploratory data anlaysis to identify any patterns and applied tokenization by ensuring we filter noise in the data and finally we applied embeddings to convert text to meaningful representation (especially for the WikiQA dataset)
- We have also implemented the word relatedness part of the Cosinet architecture as described in the pair, to allow us to apply static attention based on similarity between the question and the answer sentence words.
  
## Exploratory Data Analysis(EDA):
We used the WikiQA dataset as well as the SQuAD dataset for the task. Links for the same can be found here
https://huggingface.co/datasets/wiki_qa  
https://www.microsoft.com/en-us/download/details.aspx?id=52419
https://paperswithcode.com/dataset/squad

The datasets (after preprocessing) have been uploaded to drive and links to them have been included in the respective ipynb notebooks.

## Tokenization:
Check preprocessing.ipynb for the below steps done.

As part of tokenization, the following preprocessing processes were used:

- Remove punctuation
- Replace numbers with num tag
- Use Lower case
- Used Applied Lemmatization
- Used Applied Stemming
- Sentences were padded with < UNK >, < PAD >, < S >, < /S >
- The length of the sentences was considerably decreased after using the preceding preparation processes
- Removal of questions with no answers
- Embeddings were handles for OOV words


## Implementation of Paper 1

### 1. Word Embeddings 
Check embeds_and_cosine.ipynb for the below steps done.

As per the first paper, we used Numberbatch embeddings to create the embeddings for the words in the dataset. OOV words as well as padding, unknown and sentence tags were handled accordingly to finally create an embeddings matrix for the WikiQA dataset.

### 2. Cosinet Mechanism
Check paper1.ipynb for the below steps done.

As per the paper, we compare all the embeddings for the words in the question sentence with the embeddings of the words in the answer sentence (taking them pairwise) to compute the cosine similarity. For each word, the corresponding maximum cosine similarity is found, and the embedding for the word is extended with the coside similarity as found. This is done for all question-answer pairs, and for all the words in each of the sentences. 

## Next Steps:

### 1. Completing the Implementation of Approach 1:
- We plan to complete the implementation of the first approach by implementing the question-answer encoder, followed by a RNN model and a feed forward model to get a final score for each question-answer pair.

### 1. Implementation of Approach 2: TANDA
- In a nutshell, TANDA is a technique for fine-tuning pre-trained Transformer models sequentially in two steps:

  - first, transfer a pre-trained model to a model for a general task by fine-tuning it on a large and high-quality dataset.
  - then, perform a second fine-tuning step to adapt the transferred model to the target domain.  

- ASNQ is a dataset for answer sentence selection derived from Google Natural Questions (NQ) dataset (Kwiatkowski et al. 2019). We plan to use it to transfer the pretrained models mentioned in the above paper.

### 2. Other Base-Line Models
- We have planned to use BERT based baselines to compare the performance of the models. 
- The main model we plan to use is described in the following paper:  [__BERTSel: Answer Selection with Pre-trained models__](https://arxiv.org/pdf/1905.07588v1.pdf) 
- The code for the same can be found in the following github repository: [BERTSel](https://github.com/BPYap/BERTSel/tree/master)

- If time permits, we also intend to code another baseline model, as described in the following paper: [__BAS: An Answer Selection Method Using BERT Language Model__](https://arxiv.org/ftp/arxiv/papers/1911/1911.01528.pdf)
### 3. Comparison and Analysis
- This would include coding up the evaluation metrics as mentioned in the project outline and comparing the performance of the models agaisnt each other and against the baselines.
