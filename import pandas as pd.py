import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Setup ---
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

# 1. LOAD & PREPARE DATA
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)
df['Sentiment'] = df['label'].map({0: 'Positive', 1: 'Negative'})
df.rename(columns={'tweet': 'Comment'}, inplace=True)

# 2. OPTIMIZED CLEANING FUNCTION
def clean_text(text):
    # Remove URLs, mentions, and special characters
    text = re.sub(r"http\S+|@\S+|[^a-zA-Z\s]", "", str(text).lower())
    # Tokenize, remove stopwords, and stem
    words = [ps.stem(word) for word in text.split() if word not in STOPWORDS]
    return " ".join(words)

print("Cleaning data... please wait.")
df['Clean_Comment'] = df['Comment'].apply(clean_text)

# 3. VECTORIZATION (Using Bigrams)
# ngram_range=(1, 2) allows the model to see "not" and "good" as "not good"
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['Clean_Comment'])
y = df['Sentiment']

# 4. SPLIT & TRAIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MultinomialNB(alpha=0.1)
model.fit(X_train, y_train)

# 5. EVALUATION
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# STEP 6: ADVANCED VISUALIZATIONS
# ==========================================
plt.style.use('ggplot')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Graph A: Sentiment Distribution (Pie Chart)
df['Sentiment'].value_counts().plot(
    kind='pie', autopct='%1.1f%%', ax=axes[0, 0], colors=['#66b3ff','#ff9999'], explode=(0.05, 0.05)
)
axes[0, 0].set_title('Overall Sentiment Distribution')
axes[0, 0].set_ylabel('')

# Graph B: Confusion Matrix (Heatmap)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
axes[0, 1].set_title('Confusion Matrix: Predicted vs Actual')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

# Graph C: Word Cloud for Positive Tweets
pos_words = " ".join(df[df['Sentiment'] == 'Positive']['Clean_Comment'])
wordcloud_pos = WordCloud(width=400, height=300, background_color='white', colormap='summer').generate(pos_words)
axes[1, 0].imshow(wordcloud_pos)
axes[1, 0].set_title('Top Words in Positive Sentiment')
axes[1, 0].axis('off')

# Graph D: Word Cloud for Negative Tweets
neg_words = " ".join(df[df['Sentiment'] == 'Negative']['Clean_Comment'])
wordcloud_neg = WordCloud(width=400, height=300, background_color='white', colormap='autumn').generate(neg_words)
axes[1, 1].imshow(wordcloud_neg)
axes[1, 1].set_title('Top Words in Negative Sentiment')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 7. QUICK TEST
def quick_predict(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

print("\n--- Test Predictions ---")
print(f"Text: 'I love this!' -> Prediction: {quick_predict('I love this!')}")
print(f"Text: 'This is terrible.' -> Prediction: {quick_predict('This is terrible.')}")