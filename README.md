
# BenWise: Content Recommendation System for Media & Entertainment

Final project for the Building AI course

## Summary

BenWise is a recommendation engine driven by artificial intelligence that offers tailored content recommendations on media and entertainment channels. It generates dynamic and captivating experiences by examining user behaviour and trends, assisting users in finding articles, music, films, and television series that suit their interests.

## Background

Users find it difficult to locate relevant information on media platforms due to the abundance of digital content, which lowers engagement and increases user loss of talent, both of which limit platform growth. By providing well chosen suggestions, increasing user happiness and retention, and assisting content producers with better discoverability, 
Benwise fights this.

*Resolves media platforms' "content overload"
*Increases discoverability, which benefits content providers and
*Increases user loyality and envolvement.

## Data sources and AI methods

BenWise uses various data sources to generate recommendations, including user interaction data, content metadata, and user demographics. 

- **User Interaction Data:** Tracks user actions (clicks, views, ratings) to capture preferences.
- **Content Metadata:** Information about content, such as genre, keywords, and tags, to assess similarity.
- **User Demographics:** Basic information like age and location, which helps tailor recommendations more accurately.

Sample datasets used during development include [MovieLens]for movie recommendations and [Spotify's dataset on Kaggle] for music content. These datasets provide real-world scenarios for building and testing algorithms.

AI techniques include:
* **Collaborative Filtering:** Using user-item interactions to predict what users will enjoy.
* **Content-Based Filtering:** Leveraging content features to find similar items.
* **Hybrid Models:** Combining collaborative and content-based methods to enhance recommendation accuracy.
* **Deep Learning Models:** Autoencoders for feature extraction and Recurrent Neural Networks (RNNs) for capturing sequential patterns in user behavior.

## Challenges

The BenWise recommendation engine has certain limitations:
* **Cold Start Problem:** The system may struggle with new users or new content due to a lack of initial data.
* **Algorithmic Bias and Privacy:** Recommender systems can unintentionally reinforce biases, and handling sensitive user data requires strict privacy considerations.
* **Complex User Preferences:** Users' tastes can be dynamic, making it challenging to adapt recommendations in real-time.

Ethical considerations include avoiding filter bubbles that limit diverse content exposure and ensuring data privacy and security compliance.

## What next?

BenWise can expand by:
* **Incorporating mood-based and contextual recommendations** to enhance personalization.
* **Implementing cross-platform support** so that users can carry preferences across multiple media platforms.
* **Exploring voice-activated personalization** for smart TVs and IoT devices.

Moving forward, the project would benefit from further expertise in data engineering and NLP, as well as continuous user testing and feedback.

## Acknowledgments

* Inspired by open-source recommendation projects, including the [MovieLens dataset] and research on collaborative filtering.
* Special thanks to the University of Helsinki for the Building AI course, which inspired and guided the development of this project.
* Additional gratitude to the open-source community for providing valuable resources that have influenced this project’s development.



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

print(f'RMSE of the model: {rmse}')'''

