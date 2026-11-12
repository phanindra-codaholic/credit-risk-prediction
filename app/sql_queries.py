from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session


def fetch_prediction_stats(db: Session) -> Dict[str, Any]:
    summary_sql = text("""
        SELECT
            COUNT(*) AS total_predictions,
            COALESCE(SUM(CASE WHEN prediction = 'High Risk' THEN 1 ELSE 0 END), 0) AS high_risk_count,
            COALESCE(SUM(CASE WHEN prediction = 'Low Risk' THEN 1 ELSE 0 END), 0) AS low_risk_count,
            COALESCE(ROUND(AVG(probability), 4), 0.0) AS avg_probability,
            COALESCE(ROUND(AVG(loan_amnt), 2), 0.0) AS avg_loan_amount,
            COALESCE(ROUND(AVG(person_income), 2), 0.0) AS avg_person_income
        FROM prediction_history
    """)
    summary_row = db.execute(summary_sql).mappings().one()

    by_grade_sql = text("""
        SELECT
            loan_grade,
            COUNT(*) AS count,
            COALESCE(SUM(CASE WHEN prediction = 'High Risk' THEN 1 ELSE 0 END), 0) AS high_risk_count,
            ROUND(AVG(probability), 4) AS avg_probability
        FROM prediction_history
        GROUP BY loan_grade
        ORDER BY loan_grade ASC
    """)
    grade_rows = [dict(row) for row in db.execute(by_grade_sql).mappings().all()]

    by_intent_sql = text("""
        SELECT
            loan_intent,
            COUNT(*) AS count,
            COALESCE(SUM(CASE WHEN prediction = 'High Risk' THEN 1 ELSE 0 END), 0) AS high_risk_count,
            ROUND(AVG(probability), 4) AS avg_probability
        FROM prediction_history
        GROUP BY loan_intent
        ORDER BY count DESC, loan_intent ASC
    """)
    intent_rows = [dict(row) for row in db.execute(by_intent_sql).mappings().all()]

    total = int(summary_row["total_predictions"])
    high_risk = int(summary_row["high_risk_count"])
    high_risk_rate = round((high_risk / total) * 100.0, 2) if total > 0 else 0.0

    return {
        "total_predictions": total,
        "high_risk_count": high_risk,
        "low_risk_count": int(summary_row["low_risk_count"]),
        "high_risk_rate_pct": high_risk_rate,
        "avg_probability": float(summary_row["avg_probability"]),
        "avg_loan_amount": float(summary_row["avg_loan_amount"]),
        "avg_person_income": float(summary_row["avg_person_income"]),
        "by_loan_grade": grade_rows,
        "by_loan_intent": intent_rows,
    }


def filter_predictions_sql(
    db: Session,
    prediction: Optional[str] = None,
    loan_grade: Optional[str] = None,
    review_status: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    conditions = []
    params: Dict[str, Any] = {"limit": limit, "offset": offset}

    if prediction is not None:
        conditions.append("prediction = :prediction")
        params["prediction"] = prediction
    if loan_grade is not None:
        conditions.append("loan_grade = :loan_grade")
        params["loan_grade"] = loan_grade
    if review_status is not None:
        conditions.append("review_status = :review_status")
        params["review_status"] = review_status

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = text(f"""
        SELECT
            id, created_at, person_age, person_income, person_home_ownership,
            person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate,
            loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length,
            prediction, probability, review_status, reviewer_notes
        FROM prediction_history
        {where_clause}
        ORDER BY id DESC
        LIMIT :limit OFFSET :offset
    """)

    return [dict(row) for row in db.execute(query, params).mappings().all()]


def update_prediction_review_sql(
    db: Session,
    prediction_id: int,
    review_status: str,
    reviewer_notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    update_sql = text("""
        UPDATE prediction_history
        SET review_status = :review_status,
            reviewer_notes = :reviewer_notes
        WHERE id = :prediction_id
    """)
    result = db.execute(
        update_sql,
        {
            "prediction_id": prediction_id,
            "review_status": review_status,
            "reviewer_notes": reviewer_notes,
        },
    )
    if result.rowcount == 0:
        db.rollback()
        return None

    db.commit()

    select_sql = text("""
        SELECT
            id, created_at, person_age, person_income, person_home_ownership,
            person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate,
            loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length,
            prediction, probability, review_status, reviewer_notes
        FROM prediction_history
        WHERE id = :prediction_id
    """)
    row = db.execute(select_sql, {"prediction_id": prediction_id}).mappings().first()
    return dict(row) if row else None


def delete_prediction_sql(db: Session, prediction_id: int) -> bool:
    delete_sql = text("DELETE FROM prediction_history WHERE id = :prediction_id")
    result = db.execute(delete_sql, {"prediction_id": prediction_id})
    if result.rowcount == 0:
        db.rollback()
        return False
    db.commit()
    return True
