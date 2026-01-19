import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./water_potability.csv")

target = 'Potability'

feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']

def capping(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_cols)
    for col in X.columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = np.clip(X[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    return X

# Model Selection
model = RandomForestClassifier(
    random_state=42,
)

# Pipeline Creation
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('capping', FunctionTransformer(capping)),
    ('scaler', StandardScaler()),
    ('model', model)
])

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_dist = {
    'model__n_estimators': range(50, 501),
    'model__max_depth': range(3, 11),
    'model__max_features': range(1,10),
    'model__min_samples_leaf': range(1,6),
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=5,
    cv=3
    )
random_result = random_search.fit(X_train, y_train)

# Display results
print(f"Best Parameters: {random_result.best_params_}")
print(f"Best Score: {random_result.best_score_:.2f}")

best_model = random_result.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
bestPred = best_model.predict(X_test)
print(f"Score: {accuracy_score(y_test, bestPred):.2f}")
print(f"Precison: {precision_score(y_test, bestPred):.2f}")
print(f"Recall: {recall_score(y_test, bestPred):.2f}")
print(classification_report(y_test, bestPred))

import pickle

# Exporting
with open('./water_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

