from catboost import CatBoostRanker, Pool
import pandas as pd

df = pd.read_csv("data.csv")

df = df.sort_values("group_id").reset_index(drop=True)
  
features = df[["feat1", "feat2"]]
labels = df["relevance"]
group_id = df["group_id"]

train_pool = Pool(
    data=features,
    label=labels,
    group_id=group_id
)

model = CatBoostRanker(
    loss_function="YetiRank",    # или YetiRankPairwise, QueryRMSE, PairLogit
    iterations=500,
    learning_rate=0.1,
    depth=6
)

model.fit(train_pool)

pred = model.predict(features)
df["score"] = pred

df.sort_values(by=["group_id", "score"], ascending=[True, False])


# ИЛИ

