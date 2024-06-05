import requests
import concurrent.futures
from functools import lru_cache
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, Text, FreqDist
from collections import Counter
import string
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# Function to get all the backlinks to APA Award as this page will have the person it's given to
def get_apa_award_recipients():
    endpoint_url = "https://www.wikidata.org/w/api.php"
    parameters1 = {
        "action": "query",
        "format": "json",
        "list": "backlinks",
        "bltitle": "Q17112655",
        "blnamespace": 0,
        "bllimit": "max",
    }
    response1 = requests.get(endpoint_url, params=parameters1)
    data1 = response1.json()
    recipients = [recipient["title"] for recipient in data1["query"]["backlinks"]]
    r_ids = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_award_received, recipient): recipient for recipient in recipients}
        for future in concurrent.futures.as_completed(futures):
            recipient = futures[future]
            if future.result():
                r_ids.append(recipient)
    return r_ids

@lru_cache(maxsize=None)
def get_award_received(recipient):
    endpoint_url = "https://www.wikidata.org/w/api.php"
    parameters = {
        "action": "wbgetentities",
        "format": "json",
        "ids": recipient,
        "props": "claims",
        "languages": "en",
    }
    response = requests.get(endpoint_url, params=parameters)
    data = response.json()
    return 'P166' in data.get("entities", {}).get(recipient, {}).get("claims", {})

# Function to get wikipedia page content for given wikidata id
@lru_cache(maxsize=None)
def get_wikipedia_content(wiki_id):
    endpoint_url = "https://www.wikidata.org/w/api.php"
    parameters = {
        "action": "wbgetentities",
        "format": "json",
        "ids": wiki_id,
        "props": "labels|descriptions|claims|sitelinks",
        "sites": "enwiki",
        "languages": "en",
        "sitefilter": "enwiki"
    }
    response = requests.get(endpoint_url, params=parameters)
    data = response.json()
    if 'enwiki' in data['entities'][wiki_id]['sitelinks']:
        enwiki_title = data['entities'][wiki_id]['sitelinks']['enwiki']['title']
        wikipedia_endpoint_url = "https://en.wikipedia.org/w/api.php"
        wikipedia_parameters = {
            "action": "query",
            "format": "json",
            "titles": enwiki_title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
        }
        wikipedia_response = requests.get(wikipedia_endpoint_url, params=wikipedia_parameters)
        wikipedia_data = wikipedia_response.json()
        page_id = list(wikipedia_data["query"]["pages"].keys())[0]
        if page_id != "-1":
            content = wikipedia_data["query"]["pages"][page_id].get("extract", "")
            return {"wiki_id": wiki_id, "labels": data.get("entities", {}).get(wiki_id, {}).get("labels", {}).get("en", {}).get("value"), "descriptions": data.get("entities", {}).get(wiki_id, {}).get("descriptions", {}).get("en", {}).get("value"), "claims": data.get("entities", {}).get(wiki_id, {}).get("claims"), "intro": content, "content": content}
    return {"wiki_id": wiki_id, "labels": data.get("entities", {}).get(wiki_id, {}).get("labels", {}).get("en", {}).get("value"), "descriptions": data.get("entities", {}).get(wiki_id, {}).get("descriptions", {}).get("en", {}).get("value"), "claims": data.get("entities", {}).get(wiki_id, {}).get("claims"), "intro": None, "content": None}

# Function to get the values stored in the labels of a given Wikidata id
def get_labels(wiki_id):
    result = get_wikipedia_content(wiki_id)
    return result.get('labels')

# Function to get the values stored in the properties of a given Wikidata id
def get_data(wiki_id, p_id):
    result = []
    temp = get_wikipedia_content(wiki_id)
    temp_list = temp.get('claims', {}).get(p_id)
    if temp_list is not None:
        for i in temp_list:
            result.append(get_labels(i.get('mainsnak').get('datavalue').get('value').get('id')))
        if len(result) > 1:
            return result
        else:
            return get_labels(temp.get('claims').get(p_id)[0].get('mainsnak').get('datavalue').get('value').get('id'))

# Fetching award winners data
ids = get_apa_award_recipients()
award_winners = {}
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(get_wikipedia_content, i): i for i in ids}
    for future in concurrent.futures.as_completed(futures):
        i = futures[future]
        wiki_content = future.result()
        claims = wiki_content.get('claims', {})
        birth_date = claims.get('P569')
        if birth_date:
            birth_date = birth_date[0].get('mainsnak').get('datavalue').get('value').get('time')[1:11]
        else:
            birth_date = None
        award_winners[i] = {
            'name': get_labels(i),
            'intro': wiki_content.get('intro'),
            'gender': get_data(i, 'P21'),
            'birth_date': birth_date,
            'birth_place': get_data(i, 'P19'),
            'employer': get_data(i, 'P108'),
            'educated_at': get_data(i, 'P69'),
        }

# Filtering out the names column and then sorting it in alphabetical order
names = [award_winners[key]['name'] for key in award_winners]
sorted_names = sorted(names)
print('Award winner names in alphabetical order:')
for name in sorted_names:
    print(name)

# Function to count words in a text using word_tokenize
def count_words(text):
    words = word_tokenize(text)
    return len(words)

# Function to count sentences in a text using sent_tokenize
def count_sentences(text):
    sentences = sent_tokenize(text)
    return len(sentences)

# Function to count paragraphs in a text using the idea that wikipedia pages have new paragraphs when two newline characters are consecutive
def count_paragraphs(text):
    paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
    return len(paragraphs)

# Function to find common words after preprocessing
def common_words_preprocessed(text):
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    filtered_words = [word.translate(translator) for word in filtered_words]
    word_counts = Counter(filtered_words)
    most_common = word_counts.most_common(10)
    return ', '.join([word[0] for word in most_common])

# Using the pandas function to convert the dictionary to dataframe
df = pd.DataFrame.from_dict(award_winners, orient='index')

# Adding columns for word, sentence, and paragraph counts, and common words
df['count_words'] = df['intro'].apply(lambda x: count_words(str(x)))
df['count_sentences'] = df['intro'].apply(lambda x: count_sentences(str(x)))
df['count_paragraphs'] = df['intro'].apply(lambda x: count_paragraphs(str(x)))
df['common_words_after_preprocessing'] = df['intro'].apply(lambda x: common_words_preprocessed(str(x)))

# Display the first 10 rows of the DataFrame
award_winners_intro = df[['name','count_words','count_sentences','count_paragraphs','common_words_after_preprocessing']].copy()
award_winners_intro.head(10)

# Function to preprocess text (remove stopwords and punctuations)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    words = word_tokenize(str(text))
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    words = [word.translate(translator) for word in words]
    return words

# Extracting intro texts and preprocessing
wordslist = [preprocess_text(text) for text in df['intro']]

# Flattening the list of lists into a single list
intro_words = [word for sublist in wordslist for word in sublist]

# Print the first 10 words in the processed intro_words list
print(intro_words[:10])

# Initialize stemmers and lemmatizer
porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# Applying stemming and lemmatization
porter_stemmed_words = [porter_stemmer.stem(word) for word in intro_words]
snowball_stemmed_words = [snowball_stemmer.stem(word) for word in intro_words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in intro_words]

# Printing the first 10 results of each method
print(f'Porter Stemmer: {porter_stemmed_words[:10]}')
print(f'Snowball Stemmer: {snowball_stemmed_words[:10]}')
print(f'Lemmatizer: {lemmatized_words[:10]}')

# Initialize the VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to perform sentiment analysis using VADER
def sentiment_analysis(intro):
    return analyzer.polarity_scores(intro)['compound'] if intro else None

# Apply sentiment analysis to the 'intro' column in the DataFrame
df['sentiment'] = df['intro'].apply(sentiment_analysis)

# Display the DataFrame with sentiment scores
print(df[['name', 'sentiment']])

# Function to prepare text for LDA
def prepare_text_for_lda(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Apply the function to prepare text
df['prepared_intro'] = df['intro'].apply(lambda x: prepare_text_for_lda(str(x)) if x else '')

# Create a CountVectorizer to vectorize the text data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['prepared_intro'])

# Apply LDA for topic modeling
lda = LatentDirichletAllocation(n_components=5, random_state=0)
lda.fit(X)

# Function to display topics
def display_topics(model, feature_names, no_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        topics.append("Topic %d: %s" % (topic_idx, " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])))
    return topics

# Display the topics
topics = display_topics(lda, vectorizer.get_feature_names_out(), 10)
for topic in topics:
    print(topic)

# Function to extract named entities
def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Adding a column for named entities
df['named_entities'] = df['intro'].apply(lambda x: extract_named_entities(str(x)))

# Displaying the first 10 rows with named entities
award_winners_ner = df[['name','named_entities']].copy()
award_winners_ner.head(10)

# Plotting the distribution of word counts
plt.figure(figsize=(10, 6))
sns.histplot(df['count_words'], bins=20, kde=True)
plt.title('Distribution of Word Counts in Introductions')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Plotting the sentiment polarity
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=20, kde=True)
plt.title('Distribution of Sentiment Polarity in Introductions')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.show()