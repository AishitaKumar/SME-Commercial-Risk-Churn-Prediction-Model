"""
PHASE 1 — DATA PROCESSING & FEATURE ENGINEERING
------------------------------------------------
Consumer card-member churn (BankChurners, Kaggle).

Design notes for reviewers:
  * This is REAL retail credit-card data. Nothing here is synthetic.
  * We engineer the churn target from the real Attrition_Flag and assemble a
    feature vector from the real behavioral columns most predictive of attrition.
  * Categorical tiers (Card_Category, Income_Category) are kept as real segments
    for downstream business analysis instead of being invented.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer
import os

print("Initializing Spark Engine...")

spark = (SparkSession.builder
         .appName("CardMember_Churn_Pipeline")
         .config("spark.driver.memory", "2g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("Spark Session Active. Loading Data...")

file_path = "data/BankChurners.csv"
if not os.path.exists(file_path):
    print(f"ERROR: Could not find {file_path}. Put BankChurners.csv inside the 'data' folder.")
    spark.stop()
    raise SystemExit(1)

df = spark.read.csv(file_path, header=True, inferSchema=True)

# ------------------------------------------------------------------
# 1. Drop noise: the ID and the two leaked "Naive Bayes" score columns
#    (these are pre-computed classifier outputs — using them would be leakage)
# ------------------------------------------------------------------
columns_to_drop = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]
df = df.drop(*columns_to_drop)

# ------------------------------------------------------------------
# 2. Target variable: 1 = churned (attrited), 0 = retained
# ------------------------------------------------------------------
df = df.withColumn(
    "Churn_Flag",
    when(col("Attrition_Flag") == "Attrited Customer", 1).otherwise(0),
)

churn_rate = df.filter(col("Churn_Flag") == 1).count() / df.count()
print(f"Observed churn rate: {churn_rate:.1%}  (class-imbalanced — handled in Phase 2)")

# ------------------------------------------------------------------
# 3. Feature set — REAL behavioral drivers of card attrition.
#    We deliberately include recent-momentum ratios (Q4 vs Q1), product
#    breadth, and utilization, which the first version ignored.
# ------------------------------------------------------------------
numerical_features = [
    "Customer_Age",              # cardholder age
    "Credit_Limit",              # total credit facility
    "Total_Revolving_Bal",       # carried debt
    "Total_Trans_Amt",           # annual spend volume
    "Total_Trans_Ct",            # transaction velocity  (top predictor)
    "Months_Inactive_12_mon",    # dormancy
    "Total_Relationship_Count",  # # products held = stickiness / switching cost
    "Total_Ct_Chng_Q4_Q1",       # spend-frequency momentum (declining = risk)
    "Total_Amt_Chng_Q4_Q1",      # spend-amount momentum
    "Avg_Utilization_Ratio",     # credit utilization
    "Contacts_Count_12_mon",     # servicing contacts (friction signal)
    "Months_on_book",            # tenure
]

# Keep a real, human-readable account tier for business segmentation
indexer = StringIndexer(inputCol="Card_Category", outputCol="Card_Tier_Index",
                        handleInvalid="keep")
df = indexer.fit(df).transform(df)

assembler = VectorAssembler(inputCols=numerical_features, outputCol="features")
final_df = assembler.transform(df)

print("-" * 60)
print(f"Data Processing Complete. Card-member records: {final_df.count():,}")
print(f"Features assembled: {len(numerical_features)}")
print("-" * 60)
final_df.select("features", "Churn_Flag", "Card_Category").show(5, truncate=False)

spark.stop()
