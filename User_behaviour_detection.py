#MY SCRATCH FILE
# User_behaviour_detection


import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR =  r"C:\Siddharth\r4.2"


def parse_and_prepare(df, time_col='date'):
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    return df.dropna(subset=[time_col])

logon_df = parse_and_prepare(pd.read_csv(os.path.join(DATA_DIR, "logon.csv")), 'date')
device_df = parse_and_prepare(pd.read_csv(os.path.join(DATA_DIR, "device.csv")), 'date')
file_df = parse_and_prepare(pd.read_csv(os.path.join(DATA_DIR, "file.csv")), 'date')
email_df = parse_and_prepare(pd.read_csv(os.path.join(DATA_DIR, "email.csv")), 'date')

# Memory-safe load for large http.csv
http_df = pd.DataFrame()
for chunk in pd.read_csv(os.path.join(DATA_DIR, "http.csv"), chunksize=500_000):
    cleaned = parse_and_prepare(chunk, 'date')
    http_df = pd.concat([http_df, cleaned], ignore_index=True)


def daily_counts(df, label):
    return df.groupby(['user', df['date'].dt.date]).size().reset_index(name=label)

logon_feat = daily_counts(logon_df, "logon_count")
file_feat = daily_counts(file_df, "file_count")
email_feat = daily_counts(email_df, "email_sent")
http_feat = daily_counts(http_df, "web_visits")
device_feat = daily_counts(device_df, "usb_events")

# Merge all features
dfs = [logon_feat, file_feat, email_feat, http_feat, device_feat]
user_daily = dfs[0]
for df in dfs[1:]:
    user_daily = pd.merge(user_daily, df, on=['user', 'date'], how='outer')

user_daily.fillna(0, inplace=True)


features = user_daily.drop(['user', 'date'], axis=1)
scaled = StandardScaler().fit_transform(features)

model = IsolationForest(contamination=0.05, random_state=42)
user_daily['anomaly_score'] = model.fit_predict(scaled)
user_daily['is_anomaly'] = user_daily['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)


risk_scores = user_daily.groupby('user')['is_anomaly'].sum().reset_index()
risk_scores.columns = ['user', 'anomaly_days']
top_risky = risk_scores.sort_values(by='anomaly_days', ascending=False).head(10)


plt.figure(figsize=(10,6))
sns.barplot(data=top_risky, x='user', y='anomaly_days', palette='Reds_r')
plt.title("Top 10 Risky Users by Anomalous Days")
plt.xticks(rotation=45)
plt.ylabel("Anomalous Days")
plt.xlabel("User")
plt.tight_layout()
plt.show()

# PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
user_daily['pca1'] = pca_result[:,0]
user_daily['pca2'] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(data=user_daily, x='pca1', y='pca2', hue='is_anomaly', palette={0:'blue',1:'red'})
plt.title("User-Day Behavior PCA Anomaly View")
plt.legend(title="Anomaly")
plt.tight_layout()
plt.show()
