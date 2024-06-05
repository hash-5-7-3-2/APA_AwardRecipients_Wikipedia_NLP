# Exploring APA Award Recipients: Insights from Wikipedia Data Using NLP Techniques

This project leverages various Natural Language Processing (NLP) techniques to analyze Wikipedia data of APA Award recipients

## Data Collection

Data is collected from Wikidata and Wikipedia using their respective APIs. We extract relevant information such as names, birth dates, education, employment details and wikipedia content.

## Data Processing

The data is processed to:
- Clean and tokenize text
- Remove stopwords and punctuation
- Apply stemming and lemmatization
- Count words, sentences, and paragraphs

## Analysis

We perform several analyses on the data:
- **Common Words**: Identify the most frequent words after preprocessing.
- **Sentiment Analysis**: Assess the sentiment of the introductory sections using VADER.
- **Topic Modeling**: Identify main topics discussed in the text using Latent Dirichlet Allocation (LDA).
- **Named Entity Recognition (NER)**: Detect named entities in the text using spaCy.
