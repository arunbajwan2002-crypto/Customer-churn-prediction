# Firstly import all useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

# Making Dtatset witn some randomness.
np.random.seed(42)
n_samples = 10000

data = {
    'Customer_Id': np.arange(1000, 1000 + n_samples),
    'Tenure_months': np.random.randint(0, 61, size = n_samples).astype(int),
    'Contract_Type': np.random.choice(['Annualy', 'Quaterly', 'Monthly'], size = n_samples, p = [0.6, 0.1, 0.3]).astype(str),
    'Days_Since_Last_Activity': np.random.randint(0, 31, size = n_samples).astype(int),
    'Monthly_Spend': np.random.randint(1000, 8001, size = n_samples).astype(int),
    'Complaint_Ticket_Raised': np.random.randint(0, 9, size = n_samples).astype(int),
    'Satisfaction_Score': np.random.randint(0, 6, size = n_samples).astype(int),
    'Usage_Drop': np.random.uniform(-1, 1, size = n_samples).astype(float)
}

df = pd.DataFrame(data)
# churn formula 
churn_prob = (
    (df['Contract_Type'] == 'Monthly') * 2.5 +
    (df['Days_Since_Last_Activity'] >= 20) * 3.0 +
    (df['Complaint_Ticket_Raised'] >= 4) * 2.0 -
    (df['Satisfaction_Score'] > 4) * 2.5
)
# use sigmoid function t0 get churnprobability between 0 & 1
churn_prob = 1 / (1 + np.exp(-churn_prob))
# feature engineering
df['Churn'] = (churn_prob > np.random.uniform(0, 1, size = n_samples)).astype(int)
# one hot encoding
df_encoded = pd.get_dummies(df, columns = ['Contract_Type'], drop_first = True)

X = df_encoded.drop(columns = ['Customer_Id', 'Churn'])
y = df_encoded['Churn']
# train model witn 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

train_data = lgb.Dataset(X_train, label = y_train)
test_data = lgb.Dataset(X_test, label = y_test)


model = lgb.train(
    {'objective': 'binary', 'is_unbalance': True},
    train_data,
    num_boost_round = 100,
    valid_sets = [test_data]
)


y_pred_prob = model.predict(X_test)

Customer_threshold = 0.45

y_pred = (y_pred_prob > Customer_threshold).astype(int)



# graphs plot
plt.figure(figsize = (12, 5))

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color = 'blue', lw = 2, label = (f'roc_auc_score{roc_auc_score(y_test, y_pred):.2f}'))
plt.plot([0, 1], [0, 1], color = 'grey', linestyle = '--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend(loc = 'lower right')

# graph confusion matrix 
cm = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 1)
sns.heatmap(cm, color = 'blue', fmt = 'd', cbar = False)
plt.xlabel('Predictive Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix at Customer Threshold 0.45')
plt.show()


plt.tight_layout()
plt.savefig('Model Evaluation Matrix2.png')
plt.show()


print('--Classification Report--')
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, 'Churn_prediction_model.pkl')
print("Files saved successfully!")
