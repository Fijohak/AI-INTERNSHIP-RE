import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X = train_data[features].copy()
y = train_data["Survived"]
#预处理
X_test = test_data[features].copy()

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})

X["Age"] = X["Age"].fillna(X["Age"].median())
X_test["Age"] = X_test["Age"].fillna(X["Age"].median())

X["Fare"] = X["Fare"].fillna(X["Fare"].median())
X_test["Fare"] = X_test["Fare"].fillna(X["Fare"].median())

#划分训练集
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#模型调用
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

valid_predictions = model.predict(X_valid)
accuracy = accuracy_score(y_valid, valid_predictions)

print("验证集准确率：", accuracy)

model.fit(X, y)

test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("submission.csv 已生成")
