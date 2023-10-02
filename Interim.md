# AS2
# Answer Sentence Selection

__Team Members:__   
Kapil Rajesh,  Srija Mukhopadyay, Harinie Sivaramasethu  


## Description:
As part of the study, we investigated numerous methodologies, and this paper details the datasets we utilised, our approaches, our conclusions, and next steps. We will also go through our committed deadlines as outlined in the project overview.

## Approach:
We have performed exploratory data anlaysis to identify any patterns and applied tokenization by ensuring we filter noise in the data and finally we applied embeddings to convert text to meaningful representation.  

## Exploratory Data Analysis(EDA):
Dataset downloaded from hugging face repo originally extracted from STS Benchmark and the data available at this location: https://huggingface.co/datasets/wiki_qa

## Tokenization:
As part of tokenization, the following preprocessing processes were used:

- Remove punctuation
- Replace numbers with num tag
- Use Lower case
- Used Applied Lemmatization
- Used Applied Stemming
- Unknown treatment by replacing all words with frequency count 0 and 1 with unk tag
- The length of the sentences was considerably decreased after using the preceding preparation processes.



Maximum length of sentence in training data before preprocessing :   
Maximum length of sentence in training data after preprocessing :   

