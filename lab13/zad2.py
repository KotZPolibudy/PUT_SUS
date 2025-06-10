# Na podstawie https://marcotcr.github.io/lime/tutorials/Lime%20-%20basic%20usage%2C%20two%20class%20case.html
# Jesli chcesz wiecej wyjasnien krok po kroku, przeczytaj te strone ^^^

import lime
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.datasets


EXPLAIN_MESSAGE_NR = 100  # TODO tu zmieniaj numer klasyfikowanej wiadomosci
FEATURES_IN_EXPLANATION = 6
CLASSES = ['comp.graphics', 'sci.crypt']  # TODO wybieraj rozne pary grup z listy ponizej

#print(fetch_20newsgroups().target_names)
# 'alt.atheism',
# 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
# 'misc.forsale',
# 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
# 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
# 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'

newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset='train', categories=CLASSES)
newsgroups_test = sklearn.datasets.fetch_20newsgroups(subset='test', categories=CLASSES)


# -------- tworzymy i uczymy model --------


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def custom_tokenizer(text):
    tokens = word_tokenize(text.lower())  # małe litery
    tokens = [word for word in tokens if word.isalpha()]  # tylko słowa (bez liczb, znaków)
    tokens = [word for word in tokens if word not in stop_words]  # bez stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # lematyzacja
    return tokens

# --- Nowy vectorizer ---
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=custom_tokenizer)

# Uczenie modelu na przetworzonych danych
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rand_forest = sklearn.ensemble.RandomForestClassifier(n_estimators=300)  # zlozony klasyfikator...
rand_forest.fit(train_vectors, newsgroups_train.target)  # ...umie stworzyc skomplikowany model

pred = rand_forest.predict(test_vectors)
print('F1 score =', sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary'))


# -------- wyjasniamy decyzje modelu --------

from lime import lime_text
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline

classifier = make_pipeline(vectorizer, rand_forest)
explainer = LimeTextExplainer(class_names=CLASSES)
instance_explainer = explainer.explain_instance(
	newsgroups_test.data[EXPLAIN_MESSAGE_NR], classifier.predict_proba, num_features=FEATURES_IN_EXPLANATION)
print('Message nr = %d' % EXPLAIN_MESSAGE_NR)
print('Proba_of_class("%s") =' % CLASSES[0], classifier.predict_proba([newsgroups_test.data[EXPLAIN_MESSAGE_NR]])[0, 0])
print('Proba_of_class("%s") =' % CLASSES[1], classifier.predict_proba([newsgroups_test.data[EXPLAIN_MESSAGE_NR]])[0, 1])
print('True class = %s' % CLASSES[newsgroups_test.target[EXPLAIN_MESSAGE_NR]])

print('Explanation as feature weights =', *instance_explainer.as_list(), sep='\n\t')

fig = instance_explainer.as_pyplot_figure()
fig.savefig('explained-%s-%s-%d.png' % (CLASSES[0], CLASSES[1], EXPLAIN_MESSAGE_NR))

instance_explainer.save_to_file('explained-%s-%s-%d.html' % (CLASSES[0], CLASSES[1], EXPLAIN_MESSAGE_NR))