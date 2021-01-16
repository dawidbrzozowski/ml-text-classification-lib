# Bachelor's Thesis
This project is a part of my Bachelor's Thesis.

<br>

Project contains of:
1. Text classification library based on modern machine learning tools.
2. Django Rest Framework application used to serve model for offensive language detection.

<br>

## Project setup
---
Setup the project running the following command from the root folder:
```
source scripts/setup_project.sh
```

<br>

## Introduction
---

This project provides convenient methods to build, train, save & evaluate machine learning classifiers for text data.

<br>

### Serve model
---
You can train and serve your own models, but if you want to use the one existing in the project, created for offensive language detection in social media run the following:
```
python3 rest_api/manage.py runserver
```
This will run the server on the local host at port 8000.

The server expects to receive a json looking like this:
{
    "text": "here goes the text for offensive detection"
}
And returns a response looking like this:
{
    "offensive_rate": <float number for probability between 0-1>
}
where 0 means not offenseless text and 1 corresponds to offensive.

<br>

### Word Embeddings
---
If you wish to use methods based on GloVe Word Embeddings in the project you have to download the embeddings first.
There are two scripts for downloading two different pre-trained embeddings.
Download pre-trained embeddings, created using wikipedia data by running:
```
./scripts/download_glove_wiki.sh
```
Download pre-trained embeddings, created using twitter data by running:
```
./scripts/download_glove_twitter.sh
```
---
