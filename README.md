
# BenWise: Content Recommendation System for Media & Entertainment

Final project for the Building AI course

## Summary

BenWise is a recommendation engine driven by artificial intelligence that offers tailored content recommendations on media and entertainment channels. It generates dynamic and captivating experiences by examining user behaviour and trends, assisting users in finding articles, music, films, and television series that suit their interests.

## Background

Users find it difficult to locate relevant information on media platforms due to the abundance of digital content, which lowers engagement and increases user loss of talent, both of which limit platform growth. By providing well chosen suggestions, increasing user happiness and retention, and assisting content producers with better discoverability, Benwise fights this.
*Resolves media platforms' "content overload"
*Increases discoverability, which benefits content providers and
*Increases user loyality and envolvement.

## How is it used?

BenWise learns user preferences through interactions and interacts with media networks to provide personalised recommendations in real time. With its capacity to improve content discoverability and offer media firms useful data, it is perfect for streaming, music, and news platforms, which benefits both users and content creators.


Here’s a simple example of a code snippet to build a collaborative filtering model using SVD:

```python
# Import libraries
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load dataset
data = pd.read_csv('dataset.csv')  
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Split into training and test sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Predict and evaluate
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

print(f'RMSE of the model: {rmse}')

## Data sources and AI methods
User demographics, content metadata, and user interactions are all used by the recommendation engine. Testing is based on publicly available datasets such as MovieLens and Spotify's dataset on Kaggle. Among the AI methods are content-based filtering, collaborative filtering, and deep learning models (such as RNNs and autoencoders).

## Challenges

There are some restrictions on the BenWise recommendation engine:
* Cold Start Problem: A lack of initial data may cause the system to struggle with new users or new material.
* Algorithmic Bias and Privacy: Strict privacy considerations must be taken when managing sensitive user data, as recommender systems may inadvertently propagate biases.
* Complex User Preferences: It can be difficult to modify suggestions in real-time due to users' changing preferences.
Ethical considerations include avoiding filter bubbles that limit diverse content exposure and ensuring data privacy and security compliance.

## What next?

BenWise can grow by:
* Adding contextual and mood-based suggestions to improve personalisation.
* In order to allow consumers to carry their preferences across various media platforms, cross-platform compatibility is being implemented.
* Investigating voice-activated personalisation for Internet of Things devices and smart TVs.

Moving forward, the project would benefit from further expertise in data engineering and NLP, as well as continuous user testing and feedback.

## Acknowledgments

* Inspired by open-source recommendation projects, including the [MovieLens dataset] and research on collaborative filtering.
* Special thanks to the University of Helsinki for the Building AI course, which inspired and guided the development of this project.
* Additional gratitude to the open-source community for providing valuable resources that have influenced this project’s development.

