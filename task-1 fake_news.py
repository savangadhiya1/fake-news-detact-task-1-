import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#data
real_templates = [
    "Government announces new health policy for citizens.",
    "Scientific study shows that drinking water is good for human health.",
    "The president signed the new economy law today in the capital city.",
    "The central bank raised the interest rates to control country inflation.",
    "Scientists successfully launched a new weather satellite into orbit today."
]

fake_templates = [
    "Breaking: Aliens landed in New York yesterday night and spoke to reporters!",
    "You can win 1 million dollars by clicking this link right now for free!",
    "Eating dark chocolate every hour will make you completely invisible.",
    "NASA discovers a new planet covered entirely in solid liquid gold.",
    "Shocking: Drinking lemon juice cures all stages of cancer instantly at home!"
]

texts = []
labels = []
np.random.seed(42)

for i in range(250):
    r_txt = np.random.choice(real_templates)
    f_txt = np.random.choice(fake_templates)
    
    if np.random.rand() < 0.15:
        texts.append(r_txt)
        labels.append('FAKE')
    else:
        texts.append(r_txt)
        labels.append('REAL')
        
    if np.random.rand() < 0.15:
        texts.append(f_txt)
        labels.append('REAL')
    else:
        texts.append(f_txt)
        labels.append('FAKE')

df = pd.DataFrame({'text': texts, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#data clening
ps = PorterStemmer()
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

corpus = []
for i in range(len(df)):
    text_clean = re.sub('[^a-zA-Z]', ' ', str(df['text'][i]))
    text_clean = text_clean.lower().split()
    text_clean = [ps.stem(word) for word in text_clean if word not in stop_words]
    corpus.append(' '.join(text_clean))

#feature
tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(corpus).toarray()
Y = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model = PassiveAggressiveClassifier(max_iter=20, C=0.1, random_state=42)
model.fit(x_train, y_train)

#evalution
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, pos_label='REAL', zero_division=0) * 100
recall = recall_score(y_test, y_pred, pos_label='REAL', zero_division=0) * 100
f1 = f1_score(y_test, y_pred, pos_label='REAL', zero_division=0) * 100

#output
print("\n================ મોડેલ પર્ફોર્મન્સ રીપોર્ટ ================")
print(f'Accuracy  : {round(accuracy, 2)}%')
print(f'Precision : {round(precision, 2)}%')
print(f'Recall    : {round(recall, 2)}%')
print(f'F1-Score  : {round(f1, 2)}%')
print("=========================================================\n")

#chart
plt.figure(figsize=(7, 4))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

bars = plt.bar(metrics_names, metrics_values, color=colors, width=0.4)
plt.title('Model Performance Metrics', fontsize=12, fontweight='bold', pad=10)
plt.ylabel('Score (in %)')
plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{round(yval, 2)}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
