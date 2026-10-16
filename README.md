# 🏦 Credit Risk Prediction API

An end-to-end Machine Learning service serving a trained credit risk classification model via a high-performance **FastAPI REST API**, backed by **PostgreSQL** for prediction history auditing, containerized with **Docker & Docker Compose**, and tested with **Pytest**.

---

## 📌 1. Project Overview & Problem Statement

Financial institutions face substantial credit risk when issuing loans. Defaulted loans lead directly to capital loss, whereas rejecting creditworthy applicants forfeits interest income.

This project frames credit risk assessment as a **supervised binary classification** problem:
* **`loan_status = 0`**: Non-Default (**Low Risk**) — loan applicant is predicted to honor payment terms.
* **`loan_status = 1`**: Default (**High Risk**) — applicant demonstrates high likelihood of payment default.

The application serves the trained Scikit-learn Machine Learning pipeline in real-time through a validated REST API, persists every transaction to a relational database, and runs in an isolated container environment.

---

## 🏗️ 2. System Architecture

```text
       [ Client / Postman / Frontend ]
                     │  HTTP POST /predict
                     ▼
             [ FastAPI Application ]
                     │
         ┌───────────┴───────────┐
         │ 1. Pydantic Validation│
         └───────────┬───────────┘
                     │ Validated Input
                     ▼
         ┌───────────────────────┐
         │  Scikit-learn Pipeline│
         │  ├─ SimpleImputer     │
         │  ├─ StandardScaler    │
         │  ├─ OneHotEncoder     │
         │  └─ Random Forest     │
         └───────────┬───────────┘
                     │ Prediction & Probabilities
                     ▼
         ┌───────────────────────┐
         │  PostgreSQL Database  │  (Prediction History Log)
         └───────────┬───────────┘
                     │ HTTP 200 JSON Response
                     ▼
         { "prediction": "Low Risk", "probability": 0.1245 }
```

---

## 📊 3. Dataset & Exploratory Data Analysis (EDA)

* **Source**: Public Credit Risk Dataset (Kaggle).
* **Size**: 32,581 applicant records across 12 features.
* **Class Distribution**:
  * Non-Default (`0`): 25,473 records (**78.18%**)
  * Default (`1`): 7,108 records (**21.82%**)
* **Data Quality Discoveries**:
  * **Missing values**: `person_emp_length` (895 rows, ~2.7%), `loan_int_rate` (3,116 rows, ~9.5%).
  * **Duplicates**: 165 duplicate entries identified and pruned.
  * **Outliers**: Extreme domain entry errors discovered (`person_age` > 120 years, `person_emp_length` > 120 years) filtered during data cleaning.

---

## 🤖 4. Machine Learning & Model Comparison

Data was split using **Stratified 80/20 Train-Test Split** (`stratify=y`) to maintain class balance. Features were transformed using Scikit-Learn `ColumnTransformer` with median imputation, standard scaling, and one-hot encoding.

### Model Performance Comparison on Test Set (6,482 Samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Baseline) | 86.41% | 75.74% | 55.71% | 0.6420 | 0.8703 |
| **Decision Tree** (Max Depth=6) | 92.09% | 93.38% | 68.69% | 0.7915 | 0.8841 |
| **Random Forest** (n=100, depth=10) | **93.35%** | **97.87%** | **71.16%** | **0.8240** | **0.9252** |

* **5-Fold Stratified Cross-Validation (Random Forest)**:
  * Mean ROC-AUC: **0.9225 ± 0.0036**
* **Why Random Forest?**
  Random Forest delivered the highest Precision (97.87%) and ROC-AUC (0.9252), minimizing false positives while capturing complex non-linear interactions between applicant income, loan grade, and loan percent income.

---

## 📁 5. Repository Structure

```text
credit-risk-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, routes, and lifespan hooks
│   ├── config.py            # Environment configuration (Pydantic Settings)
│   ├── schemas.py           # Pydantic request and response models
│   ├── predictor.py         # ML pipeline loader and inference engine
│   └── database.py          # SQLAlchemy engine, session, and models
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data cleaning and ColumnTransformer definition
│   └── train.py             # Model training, comparison, CV, and export
│
├── data/
│   ├── download_data.py     # Dataset download utility
│   └── credit_risk_dataset.csv
│
├── notebooks/
│   ├── 01_eda.ipynb         # Interactive Jupyter Notebook for EDA
│   └── eda_runner.py        # Automated EDA generation script
│
├── models/
│   ├── credit_risk_pipeline.joblib   # Serialized end-to-end ML pipeline
│   └── model_metrics.json            # Model evaluation summary
│
├── tests/
│   ├── __init__.py
│   └── test_api.py          # Pytest suite testing all endpoints and validation
│
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
├── requirements.txt         # Pinned application dependencies
├── Dockerfile               # Container build instructions
├── docker-compose.yml       # Multi-container setup (FastAPI + PostgreSQL)
└── README.md                # Project documentation
```

---

## 🚀 6. Getting Started

### Local Setup (Without Docker)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/credit-risk-api.git
   cd credit-risk-api
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv myenv
   .\myenv\Scripts\activate

   # Linux / macOS
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML Pipeline (if not already generated)**:
   ```bash
   python src/train.py
   ```

5. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   Open your browser to: **http://localhost:8000/docs** for the interactive Swagger UI!

---

## 🐳 7. Running with Docker & Docker Compose

To launch the full application with **PostgreSQL** in production mode:

```bash
docker-compose up --build
```

This starts:
1. `credit_risk_db`: PostgreSQL 15 on port `5432` with persistent storage volume.
2. `credit_risk_api`: FastAPI server on port `8000` with health check monitoring.

To stop the containers:
```bash
docker-compose down
```

---

## 📡 8. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root greeting and API documentation link |
| `GET` | `/health` | Health check verifying model & database connectivity |
| `POST` | `/predict` | Evaluates credit risk, returns prediction, saves record |
| `GET` | `/predictions` | Lists paginated historical predictions from database |
| `GET` | `/predictions/{id}` | Retrieves a single prediction record by ID |
| `GET` | `/model/metrics` | Returns test-set and cross-validation evaluation metrics |

---

## 📝 9. Example Usage

### Predict Loan Default (`POST /predict`)

**Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "person_age": 30,
       "person_income": 75000,
       "person_home_ownership": "MORTGAGE",
       "person_emp_length": 5.0,
       "loan_intent": "VENTURE",
       "loan_grade": "A",
       "loan_amnt": 8000,
       "loan_int_rate": 8.9,
       "loan_percent_income": 0.11,
       "cb_person_default_on_file": "N",
       "cb_person_cred_hist_length": 6
     }'
```

**Response (HTTP 200)**:
```json
{
  "prediction": "Low Risk",
  "probability": 0.0842,
  "default_risk": "Likely Non-Default",
  "prediction_id": 1
}
```

---

## 🧪 10. Running Automated Tests

Run the complete test suite with verbose reporting:

```bash
pytest -v
```

All 9 test suites validate:
* Health checks and system availability
* Model inference and low/high risk decision boundaries
* Pydantic schema validation for illegal categories and negative values
* Database storage, retrieval, and HTTP 404 error handling

---

## 📄 11. License & Acknowledgements

* Built with [FastAPI](https://fastapi.tiangolo.com/), [Scikit-learn](https://scikit-learn.org/), and [PostgreSQL](https://www.postgresql.org/).
* Dataset originally provided by Kaggle.
