# Class-Notes-DU-AI-ML
# Introduction to Natural Language Processing (NLP)

In this lesson, we'll uncover the fundamental concepts of Natural Language Processing (NLP), providing you with a comprehensive understanding of how to prepare text data for NLP tasks. You will be introduced to key preprocessing techniques including tokenization, handling stopwords, stemming, lemmatization, and the importance of word frequency. By the end of this lesson, you will have acquired the foundational knowledge and skills to perform basic NLP tasks.

## What You'll Learn

By the end of this lesson, you will be able to:
- Define NLP and implement its workflow.
- Demonstrate how to tokenize text.
- Proficiently preprocess text, including tokenization and punctuation handling, for analysis.
- Manage and process punctuation marks and other non-alphabetic characters.
- Differentiate between stemming and lemmatization.
- Understand the importance of removing stopwords.
- Understand and demonstrate how to count tokens and n-grams.

## 1. Defining NLP and Its Workflow

**Natural Language Processing (NLP)** is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful.

### Workflow of NLP
1. **Text Acquisition**: Collecting raw text data.
2. **Text Preprocessing**: Cleaning and preparing text for analysis.
3. **Feature Extraction**: Converting text into numerical features.
4. **Modeling**: Applying machine learning models.
5. **Evaluation**: Assessing the model's performance.

**Example of NLP Workflow**:
```python
# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Step 1: Text Acquisition
text = "Natural language processing (NLP) is a field of artificial intelligence."

# Step 2: Text Preprocessing
# Tokenization
tokens = word_tokenize(text)

# Step 3: Feature Extraction (example)
# Removing stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

print(filtered_tokens)
```

## 2. Tokenization

**Tokenization** is the process of breaking down text into smaller units called tokens. These tokens can be words, sentences, or subwords.

### Word Tokenization
**Example**:
```python
from nltk.tokenize import word_tokenize

text = "Natural language processing makes it possible for computers to read text."
tokens = word_tokenize(text)
print(tokens)
```

### Sentence Tokenization
**Example**:
```python
from nltk.tokenize import sent_tokenize

text = "Natural language processing (NLP) is fascinating. It involves several techniques."
sentences = sent_tokenize(text)
print(sentences)
```

## 3. Punctuation Handling and Non-Alphabetic Characters

**Handling Punctuation and Non-Alphabetic Characters** involves cleaning the text by removing or processing these characters.

**Example**:
```python
import re

text = "Hello world! NLP is exciting, isn't it?"
# Remove punctuation
cleaned_text = re.sub(r'[^\w\s]', '', text)
print(cleaned_text)
```

## 4. Stemming and Lemmatization

### Stemming
**Stemming** is the process of reducing a word to its base or root form.

**Example**:
```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "ran", "runs", "easily", "fairly"]
stemmed_words = [stemmer.stem(word) for word in words]
print(stemmed_words)
```

### Lemmatization
**Lemmatization** reduces words to their base or root form but considers the context and converts the word to its meaningful base form.

**Example**:
```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "runs", "easily", "fairly"]
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words]
print(lemmatized_words)
```

## 5. Removing Stopwords

**Stopwords** are common words that typically do not carry significant meaning and are usually removed during text preprocessing.

**Example**:
```python
from nltk.corpus import stopwords

text = "This is a sample sentence, showing off the stop words filtration."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(text)
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
print(filtered_sentence)
```

## 6. Counting Tokens and N-Grams

### Token Counting
**Example**:
```python
from collections import Counter

text = "NLP is great. NLP is fun. NLP is useful."
tokens = word_tokenize(text)
token_counts = Counter(tokens)
print(token_counts)
```

### N-Grams
**N-Grams** are contiguous sequences of n items from a given sample of text.

**Example**:
```python
from nltk import ngrams

text = "NLP is great and NLP is fun"
tokens = word_tokenize(text)
bigrams = list(ngrams(tokens, 2))
print(bigrams)
```

### Practical Exercise

**Objective**: Apply the above concepts to preprocess and analyze a sample text.

1. **Load Text Data**: Load a sample text from a file or a predefined string.
2. **Tokenize Text**: Perform word and sentence tokenization.
3. **Clean Text**: Handle punctuation and non-alphabetic characters.
4. **Apply Stemming and Lemmatization**: Process the tokens.
5. **Remove Stopwords**: Filter out common stopwords.
6. **Count Tokens and N-Grams**: Analyze the frequency and patterns.

**Example**:
```python
# Sample Text
text = """Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of the human languages in a manner that is valuable."""

# Tokenization
tokens = word_tokenize(text)
sentences = sent_tokenize(text)

# Clean Text
cleaned_text = re.sub(r'[^\w\s]', '', text)

# Stemming
stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Lemmatization
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

# Remove Stopwords
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

# Token Counting
token_counts = Counter(filtered_tokens)

# N-Grams
bigrams = list(ngrams(filtered_tokens, 2))

print(f"Tokens: {tokens}")
print(f"Sentences: {sentences}")
print(f"Cleaned Text: {cleaned_text}")
print(f"Stemmed Tokens: {stemmed_tokens}")
print(f"Lemmatized Tokens: {lemmatized_tokens}")
print(f"Filtered Tokens: {filtered_tokens}")
print(f"Token Counts: {token_counts}")
print(f"Bigrams: {bigrams}")
```

## Conclusion

By the end of this lesson, you should have a comprehensive understanding of fundamental NLP concepts and preprocessing techniques. You will be able to prepare text data for analysis and apply basic NLP tasks efficiently using Jupyter notebooks.

### Additional Resources

- [NLTK Documentation](https://www.nltk.org/)
- [Text Preprocessing Techniques](https://towardsdatascience.com/text-preprocessing-in-nlp-29eed888ceb0)
- [Introduction to N-Grams](https://towardsdatascience.com/everything-you-need-to-know-about-n-grams-e323b38e770a)

### Exercises

1. **Exercise 1**: Apply tokenization, stemming, and lemmatization to a new text dataset.
2. **Exercise 2**: Implement a function to count the most frequent words in a given text.
3. **Exercise 3**: Create a bigram model and analyze the patterns in the text data.

By completing these exercises, you will reinforce your understanding of NLP preprocessing techniques and gain practical experience in handling text data.
