# Insider Threat Detection using Behavioral Profiling

## 📖 Overview
Insider threats are one of the hardest cybersecurity challenges because they come from individuals who already have legitimate access to enterprise systems. This project develops an **unsupervised machine learning pipeline** to detect insider threats by building behavioral profiles of users and identifying anomalies that may indicate malicious intent or risky activity.

The solution uses the **CERT Insider Threat Dataset v4.2** (Carnegie Mellon University), which simulates the activities of 1,000 users across 17 months. The dataset includes **logon, file, email, web, and device activity logs**, making it one of the most comprehensive resources for insider threat research.

---

## 🚀 Features
- **Data Aggregation**: Extracted and preprocessed log data from multiple sources into a structured user–day activity matrix.
- **Feature Engineering**: Daily counts of user activities including logons, file accesses, emails sent, web visits, and USB device events.
- **Anomaly Detection Models**: 
  - Isolation Forest (tree-based anomaly detection)
  - One-Class SVM (boundary-based detection)
  - Local Outlier Factor (density-based detection)
- **Consensus Voting**: Combined model outputs with a majority-voting system to reduce false positives and improve reliability.
- **Visualization**:
  - PCA scatter plots to show anomalies in reduced dimensions
  - Heatmaps and distributions of features
  - Risk profiling graphs for top anomalous users
- **Risk Profiling**: Generated user-level risk scores and identified the **Top 10 riskiest users**.

---

## 📊 Tech Stack
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Matplotlib-Venn  
- **Platform**: Jupyter Notebook  

---

## 📂 Dataset
The dataset used is the **CERT Insider Threat Dataset v4.2**, available for public research.  
🔗 [Download Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)

⚠️ Note: The dataset is very large and **cannot be included** in this repository. Please download it separately and update the `DATA_DIR` path in the notebook.

---

## 📌 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/insider-threat-detection.git
   cd insider-threat-detection
