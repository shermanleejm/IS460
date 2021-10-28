import pandas as pd
from sklearn.model_selection import train_test_split
import re
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from bayes_opt import BayesianOptimization
from nltk.corpus import stopwords

sw = set(stopwords.words("english"))

toxic_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train = pd.read_csv("train.csv")
evaluation = pd.read_csv("test.csv")
evaluation_labels = pd.read_csv("test_labels.csv")

model = XGBClassifier(random_state=69, seed=2, colsample_bytree=0.6, subsample=0.7)

param_grid = {
    "clf__n_estimators": [50, 100, 300],
    "clf__colsample_bytree": [0.6, 0.8, 1],
    "clf__subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
}

X = train["comment_text"].tolist()
y = train[toxic_labels].values
y = [int("".join([str(_) for _ in row]), 2) for row in y]
word_set = set()


def clean_text_and_generate_word_set(text):
    text = text.lower()
    ## remove \n \t and non-alphanumeric
    text = re.sub("(\\t|\\n)", " ", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.strip()
    ## leave 1 space between each token and remove stop words
    text_arr = []
    for x in text.split(" "):
        if len(x.strip()) > 0 and x not in sw:
            text_arr.append(x)
            word_set.add(x)
    text = " ".join(text_arr)
    return text.strip()


df = pd.DataFrame({"X": X, "y": y})
df["X"] = df["X"].apply(lambda x: clean_text_and_generate_word_set(x))

print("Starting tfidf vectorising process...")
tfidf_transformer = TfidfVectorizer()
X_train_tfidf = tfidf_transformer.fit_transform(df["X"])
feature_names = tfidf_transformer.get_feature_names()
tfidf_df = pd.DataFrame(X_train_tfidf.todense().tolist(), columns=feature_names)
tfidf_df["TARGET"] = df["y"]
print("Finished vectorising.")

## remove categories with less than 10 value counts
tfidf_df = tfidf_df.groupby("TARGET").filter(lambda x: len(x) > 10)

X_train, X_test, y_train, y_test = train_test_split(
    tfidf_df.iloc[:, tfidf_df.columns != "TARGET"],
    tfidf_df["TARGET"],
    test_size=0.20,
    random_state=69,
)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

print("Starting hyper param tuning...")
## Hyper param tuning
def xgb_eval(
    max_depth,
    min_child_weight,
    gamma,
    subsample,
    colsample_bytree,
    colsample_bylevel,
    colsample_bynode,
    reg_alpha,
    reg_lambda,
):
    params = {
        "learning_rate": 0.01,
        # "n_estimators": 10000,
        "max_depth": int(round(max_depth)),
        "min_child_weight": int(round(min_child_weight)),
        "subsample": subsample,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "colsample_bylevel": colsample_bylevel,
        "colsample_bynode": colsample_bynode,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": 69,
    }
    bst = xgb.train(params, dtrain, num_boost_round=5, verbose_eval=False)
    y_pred = bst.predict(dtest)
    res = mean_absolute_error(y_test, y_pred)
    print(res)
    return res


bopt_xgb = BayesianOptimization(
    xgb_eval,
    {
        "max_depth": (5, 15),
        "min_child_weight": (5, 80),
        "gamma": (0.2, 1),
        "subsample": (0.5, 1),
        "colsample_bytree": (0.5, 1),
        "colsample_bylevel": (0.3, 1),
        "colsample_bynode": (0.3, 1),
        "reg_alpha": (0.001, 0.3),
        "reg_lambda": (0.001, 0.3),
    },
    random_state=69,
)
bopt_xgb.maximize(n_iter=6, init_points=4)
print("Finished hyper param tuning.")

print("Training with best params...")
## best params
params = bopt_xgb.max["params"]
## convert params to int
for k in ["max_depth", "min_child_weight"]:
    params[k] = int(params[k])

model = XGBClassifier(**params)
model.fit(X_train, y_train)
print("Finished training.")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

model.save_model("model.json")
print("Saved model.")