import tweepy
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# 1. Configuración de API de Twitter
API_KEY = "TU_API_KEY"
API_KEY_SECRET = "TU_API_KEY_SECRET"
ACCESS_TOKEN = "TU_ACCESS_TOKEN"
ACCESS_TOKEN_SECRET = "TU_ACCESS_TOKEN_SECRET"

# Autenticación con Tweepy
def authenticate_twitter(api_key, api_key_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    return tweepy.API(auth)

# 2. Recolección de Tweets
def get_tweets(api, usernames, max_tweets=100):
    all_tweets = []
    for username in usernames:
        tweets = api.user_timeline(screen_name=username, count=max_tweets, tweet_mode="extended", lang="es")
        for tweet in tweets:
            all_tweets.append({"username": username, "tweet": tweet.full_text})
    return pd.DataFrame(all_tweets)

# 3. Clasificación con BERT
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def classify_tweets(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        results.append(torch.argmax(probs).item())  # 0 = No depresivo, 1 = Depresivo
    return results

# 4. Cargar Encuesta
def load_survey(file_path):
    # La encuesta debe tener columnas: "username", "depression_survey"
    return pd.read_csv(file_path)

# 5. Comparar Encuesta con Tweets
def analyze_results(tweets_df, survey_df):
    merged_df = pd.merge(tweets_df, survey_df, on="username")
    print("\n=== Resultados de Clasificación ===")
    print(classification_report(merged_df["depression_survey"], merged_df["depression_tweets"]))

    # Análisis descriptivo
    counts = merged_df["depression_tweets"].value_counts()
    plt.bar(["No depresivo", "Depresivo"], counts)
    plt.title("Distribución de Clasificaciones")
    plt.show()
    return merged_df

# 6. Flujo Principal
if __name__ == "__main__":
    # Autenticar Twitter
    api = authenticate_twitter(API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    # 6.1. Cargar datos de encuesta
    survey_file = "encuesta_gam.csv"  # Archivo CSV con "username" y "depression_survey"
    survey_df = load_survey(survey_file)

    # 6.2. Recolectar tweets
    usernames = survey_df["username"].tolist()
    tweets_df = get_tweets(api, usernames)

    # 6.3. Clasificar tweets
    tweets_df["depression_tweets"] = classify_tweets(tweets_df["tweet"].tolist())

    # 6.4. Comparar con encuesta
    results_df = analyze_results(tweets_df, survey_df)

    # 6.5. Guardar resultados
    results_df.to_csv("resultados_clasificacion.csv", index=False)
