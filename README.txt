🎭 AI Sentiment Analysis System
A complete machine learning pipeline for sentiment analysis on text reviews, with both a command-line interface and an interactive Streamlit web app.

The system supports:

Data preprocessing (tokenization, lemmatization, stopword removal)

Model training & evaluation

Prediction with confidence scores

Visualization of performance metrics

A ready-to-run web interface with interactive features

🚀 Features
Data Generation — Creates a realistic synthetic dataset of positive/negative reviews for testing.

Text Preprocessing — Cleans, tokenizes, removes stopwords, and lemmatizes text.

Model Training — Uses TF-IDF vectorization + Logistic Regression (with option for Random Forest).

Performance Evaluation — Accuracy, classification report, confusion matrix, feature importance.

Prediction — Single or batch sentiment predictions with confidence scores.

Visualizations — Confusion matrix, sentiment distribution, feature importance, prediction confidence histograms.

Web App — Fully interactive Streamlit interface for training, analysis, and visualization.

🛠️ Tech Stack
Language: Python 3.x

Libraries:

Data: pandas, numpy

NLP: nltk

ML: scikit-learn

Visualization: matplotlib, seaborn, plotly, wordcloud

Web App: streamlit

Model: Logistic Regression (default) with TF-IDF vectorization

📂 Project Structure
bash
Copy
Edit
sentiment_analyzer.py   # Main program
streamlit_app.py        # Generated Streamlit app (created at runtime)
sentiment_model.pkl     # Saved trained model
sentiment_analysis_results.png  # Generated performance plots
⚡ Quickstart
1️⃣ Install Dependencies
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn nltk streamlit plotly wordcloud
2️⃣ Download NLTK Data
python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
3️⃣ Run the CLI Demo
bash
Copy
Edit
python sentiment_analyzer.py
This will:

Generate sample data

Train the model

Show evaluation metrics

Save the model

Generate visualizations

Create the streamlit_app.py

4️⃣ Launch the Web App
bash
Copy
Edit
streamlit run streamlit_app.py
📊 Example Usage
Predict from Python:

python
Copy
Edit
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.load_model("sentiment_model.pkl")
prediction, confidence = analyzer.predict_sentiment("This movie was absolutely amazing!")
print(prediction, confidence)
🖼 Example Output
📌 Command-Line Predictions:

matlab
Copy
Edit
😊 [POSITIVE - 98.50%]: This movie is absolutely fantastic! I loved every minute of it.
😞 [NEGATIVE - 95.12%]: Terrible film with poor acting and boring plot.
😊 [POSITIVE - 73.45%]: Not bad, but could be better. Average movie overall.
📌 Visualizations (CLI generated):


Confusion Matrix showing classification accuracy

Top positive & negative words identified by the model

Distribution of confidence scores across predictions

📌 Web App (Streamlit):

(Example GIF — replace with your own)

📈 Visualizations
Confusion Matrix

Accuracy Bar

Sentiment Distribution

Top Positive/Negative Features

Confidence Histogram

Sample Prediction Confidence

🗺️ Roadmap
 Add multi-class sentiment (very positive → very negative)

 Support multilingual analysis

 Add deep learning option (e.g., BERT)

 Deploy to cloud for public access

📜 License
This project is open-source under the MIT License.