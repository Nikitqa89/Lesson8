import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
import lightgbm as lgb
from xgboost import XGBClassifier

# Загрузка данных
df_col = ['target', 'id', 'date', 'query', 'name', 'tweet']
df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                 encoding='ISO-8859-1', names=df_col, dtype={'tweet':str})

# Смотрим первые 20 строк
print(df.head(20))

# Удаляем не нужные признаки
data = df.drop(['id', 'date', 'query', 'name'], axis=1)

# Проверка пустых значений
print(data.isnull().sum())

# Проверка уникальных значений
print(data['target'].unique())

# Заменим занчения 4 на 1 для упрощения
data['target'] = data['target'].replace(4, 1)

# Приводим все в колонке tweet к нижнему регистру
data['tweet'] = data['tweet'].str.lower()

# Удаляем url-адреса
def clean_url(d):
    return re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', d)

data['tweet'] = data['tweet'].apply(lambda x: clean_url(x))

# Удаляем стоп-слова
sw = set(stopwords.words('english'))
def clean_sw(text):
    return ' '.join([word for word in str(text).split() if word not in sw])

data['tweet'] = data['tweet'].apply(lambda x: clean_sw(x))

# Удаляем все символы кроме букв
def clean_sym(d):
    return re.sub('[^a-z]', ' ', d)

data['tweet'] = data['tweet'].apply(lambda x: clean_sym(x))

# Удалим задвоенные пробелы
data['tweet'] = data['tweet'].replace(r'\s+', ' ', regex=True)


# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(data['tweet'], data['target'], test_size=0.10, random_state=42)

# Преобразуем текстовые данные в числовые признаки
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Создание и обучение модели XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

# Предсказание и оценка
y_pred = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred)
xgb_report = classification_report(y_test, y_pred)
xgb_conf_matrix = confusion_matrix(y_test, y_pred)

print("XGBClassifier Accuracy:", xgb_accuracy)
print("XGBClassifier Confusion Matrix:\n", xgb_conf_matrix)
print("XGBClassifier Classification Report:\n", xgb_report)


# Создаем и обучаем модель LGBMClassifier
lg = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=100)
lg.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = lg.predict(X_test)

# Вывод метрик классификации
lg_accuracy = accuracy_score(y_test, y_pred)
lg_report = classification_report(y_test, y_pred)
lg_conf_matrix = confusion_matrix(y_test, y_pred)

print("LGBMClassifier Accuracy:", lg_accuracy)
print("LGBMClassifier Confusion Matrix:\n", lg_conf_matrix)
print("LGBMClassifier Classification Report:\n", lg_report)


# Создаем и обучаем модель LinearSVC
svc_classifier = LinearSVC()
svc_classifier.fit(X_train, y_train)

# Сделаем предсказания на тестовом наборе данных
y_pred = svc_classifier.predict(X_test)

# Оценим качество модели
svc_accuracy = accuracy_score(y_test, y_pred)
svc_conf_matrix = confusion_matrix(y_test, y_pred)
svc_class_report = classification_report(y_test, y_pred)

print("LinearSVC Accuracy:", svc_accuracy)
print("LinearSVC Confusion Matrix:\n", svc_conf_matrix)
print("LinearSVC Classification Report:\n", svc_class_report)


# Создаем и обучаем модель BernoulliNB Naive Bayes
bnb_classifier = BernoulliNB()
bnb_classifier.fit(X_train, y_train)

# Сделаем предсказания на тестовом наборе данных
y_pred = bnb_classifier.predict(X_test)

# Оценим качество модели
bnb_accuracy = accuracy_score(y_test, y_pred)
bnb_conf_matrix = confusion_matrix(y_test, y_pred)
bnb_class_report = classification_report(y_test, y_pred)

print("BernoulliNB Accuracy:", bnb_accuracy)
print("BernoulliNB Confusion Matrix:\n", bnb_conf_matrix)
print("BernoulliNB Classification Report:\n", bnb_class_report)






