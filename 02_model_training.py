"""
PHASE 2 — MODEL TRAINING & HONEST EVALUATION
--------------------------------------------
Fixes vs the first version:
  * Evaluates ONLY on a held-out 20% test set (no more scoring on training rows).
  * Handles the 16/84 class imbalance with class weights.
  * Reports the metrics that matter for churn (Recall, Precision, F1, ROC-AUC,
    PR-AUC) plus a confusion matrix — not just accuracy on an imbalanced target.
  * Extracts REAL feature importances so the narrative is evidence-based.
  * Exports a scored file whose downstream business fields are all derived from
    REAL columns (no synthetic data).
"""

import json
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator)

print("1. Initializing Spark Engine...")
spark = (SparkSession.builder
         .appName("CardMember_Churn_Model")
         .config("spark.driver.memory", "2g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

print("2. Loading and Processing Data...")
df = spark.read.csv("data/BankChurners.csv", header=True, inferSchema=True)
df = df.drop(
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
)
df = df.withColumn("Churn_Flag",
                   when(col("Attrition_Flag") == "Attrited Customer", 1).otherwise(0))

FEATURES = [
    "Customer_Age", "Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Amt",
    "Total_Trans_Ct", "Months_Inactive_12_mon", "Total_Relationship_Count",
    "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1", "Avg_Utilization_Ratio",
    "Contacts_Count_12_mon", "Months_on_book",
]
data = VectorAssembler(inputCols=FEATURES, outputCol="features").transform(df)

# --- Class weights: balance the 16% churn / 84% retain split ---
n = data.count()
pos = data.filter(col("Churn_Flag") == 1).count()
w_pos = n / (2.0 * pos)
w_neg = n / (2.0 * (n - pos))
data = data.withColumn("weight",
                       when(col("Churn_Flag") == 1, lit(w_pos)).otherwise(lit(w_neg)))

print("3. Splitting into Train (80%) / Test (20%)...")
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

print("4. Training the Random Forest Classifier (class-weighted)...")
rf = RandomForestClassifier(
    featuresCol="features", labelCol="Churn_Flag", weightCol="weight",
    numTrees=150, maxDepth=8, seed=42,
)
rf_model = rf.fit(train_data)

print("5. Evaluating on the HELD-OUT TEST SET (data the model never saw)...")
pred = rf_model.transform(test_data)

auc = BinaryClassificationEvaluator(labelCol="Churn_Flag",
                                    metricName="areaUnderROC").evaluate(pred)
prauc = BinaryClassificationEvaluator(labelCol="Churn_Flag",
                                      metricName="areaUnderPR").evaluate(pred)
acc = MulticlassClassificationEvaluator(labelCol="Churn_Flag",
                                        metricName="accuracy").evaluate(pred)
f1 = MulticlassClassificationEvaluator(labelCol="Churn_Flag",
                                       metricName="f1").evaluate(pred)

tp = pred.filter((col("Churn_Flag") == 1) & (col("prediction") == 1)).count()
fp = pred.filter((col("Churn_Flag") == 0) & (col("prediction") == 1)).count()
fn = pred.filter((col("Churn_Flag") == 1) & (col("prediction") == 0)).count()
tn = pred.filter((col("Churn_Flag") == 0) & (col("prediction") == 0)).count()
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0

print("-" * 60)
print("MODEL PERFORMANCE (held-out test set)")
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  PR-AUC    : {prauc:.4f}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  Precision : {precision:.4f}  (of flagged churners, how many really churned)")
print(f"  Recall    : {recall:.4f}  (of real churners, how many we caught)")
print(f"  Confusion : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print("-" * 60)

print("6. Extracting REAL feature importances...")
importances = sorted(zip(FEATURES, rf_model.featureImportances.toArray()),
                     key=lambda x: -x[1])
for name, val in importances:
    print(f"    {name:26s} {val:.4f}")
top_feature = importances[0][0]

# Persist metrics so the presentation reads real numbers instead of hardcoding
metrics = {
    "roc_auc": round(auc, 4), "pr_auc": round(prauc, 4),
    "accuracy": round(acc, 4), "f1": round(f1, 4),
    "precision": round(precision, 4), "recall": round(recall, 4),
    "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    "top_features": [{"feature": f, "importance": round(float(v), 4)}
                     for f, v in importances[:5]],
    "test_set_size": tp + fp + fn + tn,
}
with open("data/model_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
print("   -> saved data/model_metrics.json")

# ------------------------------------------------------------------
# 7. Score the FULL population and export business-ready fields.
#    Every exported column below is derived from REAL data.
# ------------------------------------------------------------------
print("7. Scoring full population and exporting for the dashboard...")
scored = rf_model.transform(data)

export = scored.select(
    "Customer_Age", "Credit_Limit", "Total_Revolving_Bal", "Total_Trans_Amt",
    "Total_Trans_Ct", "Months_Inactive_12_mon", "Total_Relationship_Count",
    "Total_Ct_Chng_Q4_Q1", "Total_Amt_Chng_Q4_Q1", "Avg_Utilization_Ratio",
    "Contacts_Count_12_mon", "Months_on_book", "Card_Category",
    "Churn_Flag", "prediction", "probability",
).toPandas()

# Clean probability -> churn risk score (P(churn))
export["Churn_Risk_Score"] = export["probability"].apply(lambda v: float(v[1]))
export = export.drop(columns=["probability"])

# Priority score = risk x real annual spend, min-max scaled to 0-100
export["Expected_Loss"] = export["Churn_Risk_Score"] * export["Total_Trans_Amt"]
export["Priority_Score"] = (
    export["Expected_Loss"] / export["Expected_Loss"].max() * 100
).round(1)

export = export.sort_values("Priority_Score", ascending=False)
export.to_csv("data/SME_Churn_Predictions_Prioritized.csv", index=False)
print(f"   -> saved data/SME_Churn_Predictions_Prioritized.csv ({len(export):,} rows)")
print(f"\nHeadline for slides: Recall {recall:.0%} at ROC-AUC {auc:.3f}; "
      f"top predictor = {top_feature}.")

spark.stop()
