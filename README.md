# Customer Sentiment Analysis (NLP)

## Overview
This project analyzes customer reviews to predict sentiment (positive/negative). Using natural language processing (NLP) and machine learning, the model achieved 82% accuracy, providing actionable insights for product improvement and enhanced customer experience.

## Features
- Load and clean text data
- Preprocess using TF-IDF vectorization
- Train classifiers:
  - Logistic Regression
  - Multinomial Naive Bayes
- Evaluate model using accuracy, confusion matrix, and classification report
- Generate actionable insights from customer sentiment

##Results

Achieved 82% accuracy on the test set

Insights allow better understanding of customer feedback

Can support product and service improvement decisions

## Usage
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
Place your dataset in data/reviews.csv (columns: review, sentiment).

Run the notebook notebooks/sentiment_analysis.ipynb to train and evaluate the model.


