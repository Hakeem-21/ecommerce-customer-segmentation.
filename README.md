# End-to-End E-Commerce Customer Segmentation (RFM Analysis & K-Means)

An end-to-end unsupervised machine learning project that segments online retail customers using **RFM (Recency, Frequency, Monetary)** feature engineering and **K-Means clustering**. The project processes over 500,000 transaction records into distinct, actionable customer personas to drive targeted marketing and retention strategies.

---

## 📌 Executive Summary

* **Dataset:** 541,909 raw transaction records from a UK-based online retail platform (Dec 2010 – Dec 2011).
* **Final Customer Base:** 4,338 unique, verified customer profiles.
* **Core Objective:** Overcome high feature skewness and transactional noise to segment customers into actionable tiers using unsupervised clustering.
* **Outcome:** Identified 3 primary behavioral segments (**VIP / Champions**, **Active Loyalists**, and **Dormant / At-Risk**), balancing mathematical clustering metrics with business utility.

---

## 🏗️ Architecture & Technical Pipeline

```text
Raw CSV Data (Transactions)
   │
   ▼
Data Cleaning & Filtering
   ├── Filter missing CustomerIDs (~135k unidentifiable records removed)
   ├── Strip return/cancellation orders ('C' prefix in InvoiceNo)
   └── Vectorized datetime parsing with strict format matching
   │
   ▼
RFM Feature Engineering
   ├── Snapshot Reference Date: max(InvoiceDate) + 1 Day (2011-12-10)
   ├── Recency: Days since last transaction (lambda x: (snapshot - max(x)).days)
   ├── Frequency: Count of distinct orders (nunique on InvoiceNo)
   └── Monetary: Total revenue per customer (Quantity * UnitPrice)
   │
   ▼
Statistical Normalization
   ├── Skewness Diagnosis: Frequency (+12.07) & Monetary (+19.34)
   ├── Logarithmic Transformation (np.log1p / np.log) -> Skewness normalized to [-0.38, 0.78]
   └── StandardScaler: Mean = 0.00, Std = 1.00
   │
   ▼
K-Means Clustering & Model Selection
   ├── Evaluated k in [2, 10] using Inertia (Elbow Method) & Silhouette Scores
   └── Selected k = 3 based on business utility and distinct behavioral profiles
   │
   ▼
Cluster Profiling & Data Export
   ├── Mean & Median behavioral analysis
   └── Exported customer profiles to ecommerce_customer_segments.csv
```

---

## 📊 Customer Segment Profiles & Strategic Playbook

| Segment | Behavioral Profile | Business Focus | Marketing Action |
| :--- | :--- | :--- | :--- |
| **Champions / VIPs** | Low Recency, High Frequency, High Total Spend | Retention & Advocacy | Early access to new products, exclusive VIP rewards, zero discount dilution. |
| **Potential Loyalists** | Moderate Recency, Moderate Frequency, Steady Spend | Upselling & Basket Size | Cross-sell complementary categories, free shipping thresholds, loyalty tier onboarding. |
| **Dormant / At-Risk** | High Recency (inactive for months), Low Frequency | Reactivation / Cost Optimization | Re-engagement discount codes, exit surveys, suppress from high-cost paid ad targeting. |

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.11.17
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`KMeans`, `StandardScaler`, `silhouette_score`)
* **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hakeem-21/ecommerce-customer-segmentation.git
   cd ecommerce-customer-segmentation
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. **Launch the Notebook:**
   ```bash
   jupyter notebook customer_segmentation.ipynb
   ```

---

## 📂 Repository Structure

```text
├── data/
│   └── online_retail_II.csv                         # Raw retail transaction dataset
├── customer_segmentation.ipynb          # End-to-end Python analysis notebook
├── ecommerce_customer_segments.csv      # Exported customer table with cluster labels
├── README.md                            # Project documentation & business summary
└── requirements.txt                     # Python package dependencies
```