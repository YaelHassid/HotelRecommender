import os
from flask.cli import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import ast
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pymongo import MongoClient
import certifi

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("HotelRecommendation") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.shuffle.file.buffer", "32k") \
    .config("spark.shuffle.memoryFraction", "0.3") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.rpc.message.maxSize", "256") \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.network.timeout", "600s") \
    .getOrCreate()

# Load the Dataset
reviews_file_path = 'archive/reviews.csv'
pandas_reviews_df = pd.read_csv(reviews_file_path)
pandas_reviews_df = pandas_reviews_df[['ratings', 'author', 'offering_id']]
print("loaded")

# Preprocessing
def calculate_mean(ratings):
    try:
        ratings_dict = ast.literal_eval(ratings)
        values = [float(v) for v in ratings_dict.values() if isinstance(v, (int, float))]
        if not values:
            return None
        return sum(values) / len(values)
    except (ValueError, SyntaxError, TypeError):
        return None

# Register the UDF
mean_rating_udf = udf(calculate_mean, DoubleType())

# Convert pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(pandas_reviews_df)
spark_df = spark_df.repartition(200)

print("converted to spark")
# Apply the UDF to the DataFrame
spark_df = spark_df.withColumn('mean_rating', mean_rating_udf(col('ratings')))

def extract_username(author):
    try:
        author_dict = ast.literal_eval(author)
        return author_dict.get('username', None)
    except (ValueError, SyntaxError, TypeError):
        return None

# Register the UDF
username_udf = udf(extract_username, StringType())

# Apply the UDF to the DataFrame
spark_df = spark_df.withColumn('username', username_udf(col('author')))

# Drop unnecessary columns
spark_df = spark_df.drop('ratings', 'author')

# Function to assign numeric IDs to usernames and keep the username column
def assign_numeric_ids(spark_df, id_column="username"):
    # Create a DataFrame with distinct usernames and corresponding numeric IDs
    unique_usernames = spark_df.select(id_column).distinct().withColumn("numeric_id", monotonically_increasing_id())
    
    # Join with the original DataFrame to add the numeric_id column
    spark_df_with_ids = spark_df.join(unique_usernames, on=id_column, how="left")
    
    return spark_df_with_ids

# Apply numeric IDs to usernames
spark_df = assign_numeric_ids(spark_df)

spark_df = spark_df.withColumn("numeric_id", col("numeric_id").cast("int"))


spark_df.show()
# Cache the DataFrame to optimize performance
spark_df.cache()

# Load environment variables
load_dotenv()

# MongoDB connection settings
mongo_uri = os.getenv('MONGO_URI')
mongo_db = "Hotel_Recommendation"
mongo_collection = "review_with_index"

# Use certifi to get the path to the CA bundle
ca = certifi.where()

# Initialize the MongoClient with TLS settings
client = MongoClient(
    mongo_uri,
    tls=True,  # Enable TLS
    tlsCAFile=ca  # Path to CA bundle
)

# Access the database and collection
db = client[mongo_db]
collection = db[mongo_collection]

# Convert Spark DataFrame to Pandas DataFrame for MongoDB upload
pandas_df = spark_df.toPandas()

# Function to insert data in chunks
def insert_data_in_chunks(data, chunk_size=1000):
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        collection.insert_many(chunk)

# Insert data in chunks
insert_data_in_chunks(pandas_df.to_dict("records"))
print("Data uploaded to MongoDB successfully.")

# Model Building, Training & Testing

# Prepare ALS data
als_data = spark_df.select(col("numeric_id").alias("user_id"),
                            col("offering_id").alias("item_id"),
                            col("mean_rating").alias("rating"))

# Initialize the ALS model
als = ALS(userCol="user_id",
          itemCol="item_id",
          ratingCol="rating",
          coldStartStrategy="drop")

# Define the parameter grid for Cross-Validation
param_grid = (ParamGridBuilder()
               .addGrid(als.rank, [5, 10, 15])
               .addGrid(als.maxIter, [5, 10])
               .addGrid(als.regParam, [0.01, 0.1, 0.2])
               .build())

# Define the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# Initialize CrossValidator
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=3,  # Adjust the number of folds as needed
                          parallelism=4)  # Increase parallelism if resources allow

# Split the data into training and testing sets
(training_data, test_data) = als_data.randomSplit([0.8, 0.2])

# Fit the CrossValidator model
cv_model = crossval.fit(training_data)

# Make predictions on the test data
predictions = cv_model.transform(test_data)

# Evaluate the model
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Generate top 10 restaurant recommendations for each customer using the best model
best_als_model = cv_model.bestModel
user_recs = best_als_model.recommendForAllUsers(10)
user_recs.show(truncate=False)

# Save the Model
model_path = "models/als_model"
best_als_model.write().overwrite().save(model_path)
print("Model saved successfully.")

# Stop the Spark session
spark.stop()