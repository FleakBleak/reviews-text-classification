# Text Classification for Business

This project focuses on building a text classification model to analyze sentiment in e-commerce customer reviews, specifically in the domain of women's clothing. The project uses machine learning techniques and a variety of preprocessing steps to classify customer reviews into positive or negative sentiments.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results and Discussion](#results-and-discussion)
- [Conclusion](#conclusion)
- [Limitations](#limitations)

## Introduction
In the rapidly growing field of e-commerce, understanding customer sentiment is crucial. This project aims to classify customer reviews as positive or negative using machine learning algorithms like Naive Bayes and Logistic Regression. Key preprocessing steps include tokenization, stop words removal, and lemmatization. Feature extraction is done using TF-IDF vectorization.

## Methodology
The project uses the following methodology:

1. **Data Preparation**: Collection and preprocessing of e-commerce review data.
2. **Data Preprocessing**: Techniques such as tokenization, stop words removal, lemmatization, and TF-IDF vectorization.
3. **Machine Learning Models**: Training and evaluation of Logistic Regression and Naive Bayes classifiers.
4. **Evaluation**: Metrics like confusion matrix, precision, recall, F1-score, and ROC curve are used for evaluation.

## Results and Discussion
Logistic Regression performed better than Naive Bayes for sentiment analysis on customer reviews. However, there was a bias towards the majority class due to class imbalance. Hyperparameter tuning helped in reducing bias. The ROC curve showed a higher AUC for Logistic Regression compared to Naive Bayes.

## Conclusion
The project demonstrated the effectiveness of text classification for business applications, specifically for sentiment analysis of customer reviews. The model can help businesses gain insights into customer satisfaction and make data-driven decisions to improve products or services.

## Limitations:
- Accuracy depends on the quality and quantity of training data.
- The model may struggle with sarcasm, irony, or context-dependent text.
- Handling of misspelled or uncommon words may be limited.
