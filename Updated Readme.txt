from optimize_parameters import *
from text_preprocessing import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from hyperopt import hp
from bayesian_search import bayesian_search
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
import datetime

# Initialize Spark session with Delta Lake support
builder = SparkSession.builder.appName("DeltaLakeExample") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:1.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

def save_to_delta(df, s3_path):
    # Convert Pandas DataFrame to Spark DataFrame
    spark_df = spark.createDataFrame(df)
    
    # Write Spark DataFrame to Delta table
    spark_df.write.format("delta").mode("overwrite").save(s3_path)

# Load email data
df_emails = pd.read_csv(
    "/mnt/nfsdata/vikranthc/gitlab/email-sentiment-reporting/code/topicmodelling_angryemails/20240630_angryemails.csv",
    encoding="utf8"
)

# Clean email data
df_emails["cleaned_subject"] = df_emails["subject"].apply(clean_subject)
df_emails["cleaned_body"] = df_emails["body_content"].apply(clean_body)
df_emails["combined_text"] = df_emails["cleaned_subject"] + " " + df_emails["cleaned_body"]
df_emails.drop_duplicates(subset=["combined_text"], keep="first", inplace=True)
data_corpus = df_emails["combined_text"].tolist()

# Sentence embeddings and keyword extraction
sentence_model_march = SentenceTransformer("all-mpnet-base-v2")
embeddings = sentence_model_march.encode(data_corpus, show_progress_bar=True)

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(data_corpus)

vocabulary = [k[0] for keyword in keywords for k in keyword]
vocabulary = list(set(vocabulary))

# Vectorization and transformation
stopwords = list(stopwords.words("english"))
vectorizer_model = CountVectorizer(vocabulary=vocabulary, stop_words=stopwords)
representation_model = MaximalMarginalRelevance(diversity=0.5)
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Hyperparameter space for optimization
hspace = {
    "n_neighbors": hp.choice("n_neighbors", range(3, 30)),
    "n_components": hp.choice("n_components", range(3, 30)),
    "min_cluster_size": hp.choice("min_cluster_size", range(20, 150)),
    "random_state": 42,
}

label_lower = 8
label_upper = 25

# Bayesian search
optimized_params = bayesian_search(hspace, data_corpus, embeddings, vectorizer_model, ctfidf_model, UMAP, HDBSCAN, label_lower, label_upper)

umap_model = UMAP(n_neighbors=optimized_params["n_neighbors"], n_components=optimized_params["n_components"], random_state=optimized_params["random_state"])
hdbscan_model = HDBSCAN(min_cluster_size=optimized_params["min_cluster_size"])

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    calculate_probabilities=True,
    verbose=True
)

topics, probabilities = topic_model.fit_transform(data_corpus, embeddings)

# Extract topic information
topic_info = topic_model.get_topic_info()
topic_info["Count"] = topic_info["Count"].astype(int)

# Save the results to Delta Lake table in S3
year = datetime.datetime.now().year
month = datetime.datetime.now().month
day = datetime.datetime.now().day
s3_path = f"s3a://your-s3-bucket/Dev/angry_emails_analysis/{year}/{month}/{day}"

save_to_delta(topic_info, s3_path)
