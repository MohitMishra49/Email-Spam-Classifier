# 📧 Email Spam Classifier

A machine learning web app that classifies a message as **Spam** or **Not Spam (Ham)** using a Naive Bayes classifier trained on a labeled SMS/email dataset. Built with Python, Scikit-learn, and NLTK, and served through a Streamlit interface.

**Live Demo:** [email-frontend-dl1y.onrender.com](https://email-frontend-dl1y.onrender.com/)

---

## Overview

The classifier is trained on a dataset of 5,572 labeled messages (ham/spam). After cleaning and de-duplication (5,169 unique messages remained), the text is normalized through a custom NLP pipeline and converted into numerical features, which are then fed into a Naive Bayes model for classification.

| Metric | Score |
|---|---|
| Accuracy | 97.0% |
| Precision | 97.3% |

Precision was prioritized over recall/accuracy alone, since minimizing false positives (a real message wrongly flagged as spam) matters more than catching every last spam message.

## How It Works

**1. Preprocessing (`transform_text`)**
Each message is cleaned before vectorization:
- Lowercased
- Tokenized with `nltk.word_tokenize`
- Non-alphanumeric tokens removed
- Stopwords and punctuation removed (`nltk.corpus.stopwords`)
- Stemmed using NLTK's `PorterStemmer`

```python
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    text = y[:]
    y.clear()
    y = [i for i in text if i not in stopwords.words("english") and i not in string.punctuation]

    text = y[:]
    y.clear()
    y = [ps.stem(i) for i in text]

    return " ".join(y)
```

**2. Feature Extraction**
Cleaned text is vectorized into a bag-of-words / TF-IDF representation before being passed to the model.

**3. Model Selection**
Three Naive Bayes variants were trained and compared on the same train/test split (80/20):

| Model | Accuracy | Precision |
|---|---|---|
| Gaussian NB | ~88.0% | ~53.2% |
| Multinomial NB | ~96.4% | — |
| **Bernoulli NB (final)** | **97.0%** | **97.3%** |

Bernoulli Naive Bayes gave the best precision, so it was selected as the final model and serialized for deployment.

**4. Prediction**
The trained model (`model.pkl`) and fitted vectorizer (`vectorizer.pkl`) are loaded at runtime. User input is transformed, vectorized, and classified as spam (1) or not spam (0).

## Tech Stack

- **Language:** Python
- **ML/NLP:** Scikit-learn, NLTK
- **App/UI:** Streamlit
- **Data handling:** Pandas, NumPy
- **Deployment:** Render / Heroku-style Procfile

## Project Structure

```
Email-Spam-Classifier/
├── email.ipynb          # EDA, preprocessing, model training & comparison
├── mail.py               # Streamlit app (loads model + vectorizer, serves predictions)
├── model.pkl              # Trained Bernoulli Naive Bayes model
├── vectorizer.pkl          # Fitted vectorizer
├── requirements.txt      # Python dependencies
├── nltk.txt               # NLTK corpora required at deploy time (stopwords, punkt)
├── setup.sh               # Streamlit server config for deployment
└── procfile               # Process definition for deployment
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/MohitMishra49/Email-Spam-Classifier.git
cd Email-Spam-Classifier
pip install -r requirements.txt
```

### Download required NLTK data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Run locally

```bash
streamlit run mail.py
```

The app will open in your browser. Paste any message into the text box and click **Predict** to see whether it's classified as spam or not.

## Model Training

To retrain or explore the model from scratch, open `email.ipynb`. It walks through:
- Loading and cleaning the raw dataset
- Exploratory data analysis (class balance, message length/word/sentence distributions)
- Text preprocessing and stemming
- Vectorization
- Training and evaluating Gaussian, Multinomial, and Bernoulli Naive Bayes models
- Exporting the final model and vectorizer as `.pkl` files

## Future Improvements

- Address class imbalance (spam messages are ~13% of the dataset) with resampling or class weighting
- Experiment with additional models (Logistic Regression, SVM, ensemble methods)
- Add unit tests around the preprocessing pipeline
- Expose predictions via a REST API in addition to the Streamlit UI

## Author

**Mohit Mishra**
[GitHub](https://github.com/MohitMishra49) · [LinkedIn](https://linkedin.com/in/mohitmishra49) · [Hugging Face](https://huggingface.co/MohitMishra49)
