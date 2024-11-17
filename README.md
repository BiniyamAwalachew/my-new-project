
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

print(f'RMSE of the model: {rmse}')'''

## Data sources and AI methods

The recommendation engine in this project utilizes various data sources to generate personalized content suggestions, including:

- **User Demographics:** Information such as age, gender, and location, which can help tailor recommendations more accurately.
- **Content Metadata:** Details about the content being recommended, such as genre, keywords, and descriptions.
- **User Interactions:** Data from user behaviors such as clicks, views, ratings, and watch history, which help the system learn user preferences.

For testing the recommendation engine, publicly available datasets such as the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and [Spotify's dataset on Kaggle](https://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db) have been used. These datasets provide rich, real-world examples of user interactions with media content, which are essential for training and validating the model.

### AI Techniques Used:
- **Collaborative Filtering:** A method that relies on user-item interactions to make predictions about what a user may like based on the preferences of similar users.
- **Content-Based Filtering:** A technique that recommends items similar to those the user has shown interest in, based on the content's attributes (e.g., genre, keywords).
- **Deep Learning Models:** Models like Recurrent Neural Networks (RNNs) and Autoencoders, which are used for more complex recommendations, such as predicting user preferences over time or learning latent features from the data.

## Challenges

The **BenWise** recommendation engine, like any AI system, has limitations and challenges that need to be addressed:

- **Cold Start Problem:** The system may struggle to provide accurate recommendations for new users or new items due to the lack of historical data.
- **Algorithmic Bias and Privacy:** Recommendation engines can inadvertently propagate biases, such as reinforcing stereotypes or showing content from certain categories more frequently. Additionally, sensitive user data must be handled with care to ensure privacy and compliance with data protection regulations.
- **Complex User Preferences:** Users' preferences are often dynamic and context-dependent, which makes it challenging for the system to adapt in real-time.

Ethical considerations also play a key role, including ensuring that the system does not create "filter bubbles" by limiting content diversity and that users' privacy is respected in all interactions.

## What next?

To further enhance **BenWise**, the following features could be added:

- **Mood-Based and Contextual Recommendations:** By analyzing the user's mood or context (e.g., time of day, device used), recommendations could be made even more personalized.
- **Cross-Platform Support:** The ability to carry over user preferences across multiple platforms (e.g., from mobile to desktop) could improve the user experience.
- **Voice-Activated Personalization:** Implementing voice control for recommendations, especially for smart TVs and IoT devices, could provide a hands-free, seamless experience for users.

In order to continue evolving this project, further expertise in **data engineering**, **natural language processing (NLP)**, and **user testing** would be beneficial. Continuous feedback from real-world usage would also help to refine the recommendation engine.

## Acknowledgments

- Inspired by various open-source recommendation systems, including the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and collaborative filtering research.
- Special thanks to the **University of Helsinki** for the Building AI course, which provided the knowledge and guidance needed to develop this project.
- Grateful to the open-source community for the tools, libraries, and datasets that made this project possible.

