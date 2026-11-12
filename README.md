# Credit Risk Prediction API

Machine learning service that predicts loan default risk (`Low Risk` / `High Risk`) from applicant financial and demographic attributes, served via FastAPI and backed by a SQL database for prediction history and risk analytics.

## Overview

- **Dataset**: Public Kaggle Credit Risk Dataset (32,581 records, 12 columns; 32,409 after dropping duplicates and age/employment outliers > 100).
- **Target (`loan_status`)**: `0` = Non-Default (78.18%), `1` = Default (21.82%).
- **Preprocessing**: `ColumnTransformer` with median imputation + `StandardScaler` for numerical features and most-frequent imputation + `OneHotEncoder` for categorical features, bundled with the classifier in a Scikit-learn `Pipeline`.

## Model Evaluation (20% Stratified Test Split, 6,482 Samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 86.41% | 75.74% | 55.71% | 0.6420 | 0.8703 |
| Decision Tree (max_depth=6) | 92.09% | 93.38% | 68.69% | 0.7915 | 0.8841 |
| Random Forest (n_estimators=100, max_depth=10) | 93.35% | 97.87% | 71.16% | 0.8240 | 0.9252 |

- 5-fold stratified cross-validation (Random Forest) mean ROC-AUC: `0.9225` (`+/- 0.0036`).

## Project Structure

```text
├── app/
│   ├── config.py          # App & DB settings (.env)
│   ├── database.py        # SQLAlchemy engine, PredictionHistory model, schema setup
│   ├── main.py            # FastAPI routes and lifespan startup
│   ├── predictor.py       # Loads serialized pipeline and runs inference
│   ├── schemas.py         # Pydantic request/response schemas
│   └── sql_queries.py     # Parameterized SQL queries (filtering, stats, updates)
├── data/
│   ├── download_data.py
│   └── credit_risk_dataset.csv
├── models/
│   ├── credit_risk_pipeline.joblib
│   └── model_metrics.json
├── src/
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   └── test_api.py
├── .env.example
├── .gitignore
├── main.py
└── requirements.txt
```

## Setup & Usage

1. Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   # Windows: .\myenv\Scripts\activate
   # Linux/macOS: source myenv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables (optional; defaults to local SQLite `credit_risk.db`):
   ```bash
   cp .env.example .env
   ```
   To use PostgreSQL instead of SQLite, set `DATABASE_URL` in `.env` (for example `postgresql://postgres:password@localhost:5432/credit_risk_db`).

4. Train the model pipeline (optional if `models/credit_risk_pipeline.joblib` already exists):
   ```bash
   python src/train.py
   ```

5. Start the API server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   Interactive API docs are available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root status response |
| `GET` | `/health` | Database and model health check |
| `POST` | `/predict` | Validate input, run inference, log record to SQL |
| `GET` | `/predictions` | Query prediction history (`prediction`, `loan_grade`, `review_status`, `limit`, `offset`) |
| `GET` | `/predictions/stats` | Portfolio risk summary and `GROUP BY` breakdowns |
| `GET` | `/predictions/{id}` | Fetch a single prediction record |
| `PATCH` | `/predictions/{id}` | Update `review_status` (`PENDING`, `APPROVED`, `REJECTED`) and notes |
| `DELETE` | `/predictions/{id}` | Delete a prediction record |
| `GET` | `/model/metrics` | Return saved model evaluation metrics |

## Example Request

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

## Tests

```bash
pytest -v
```
