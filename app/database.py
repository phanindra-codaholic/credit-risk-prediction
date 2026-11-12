from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from app.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    person_age = Column(Integer, nullable=False)
    person_income = Column(Float, nullable=False)
    person_home_ownership = Column(String(50), nullable=False)
    person_emp_length = Column(Float, nullable=True)
    loan_intent = Column(String(50), nullable=False)
    loan_grade = Column(String(10), nullable=False, index=True)
    loan_amnt = Column(Float, nullable=False)
    loan_int_rate = Column(Float, nullable=True)
    loan_percent_income = Column(Float, nullable=False)
    cb_person_default_on_file = Column(String(5), nullable=False)
    cb_person_cred_hist_length = Column(Integer, nullable=False)

    prediction = Column(String(20), nullable=False, index=True)
    probability = Column(Float, nullable=False)
    review_status = Column(String(20), nullable=False, default="PENDING", server_default="PENDING", index=True)
    reviewer_notes = Column(Text, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if "prediction_history" in inspector.get_table_names():
        existing_cols = {col["name"] for col in inspector.get_columns("prediction_history")}
        with engine.begin() as conn:
            if "review_status" not in existing_cols:
                conn.execute(
                    text("ALTER TABLE prediction_history ADD COLUMN review_status VARCHAR(20) NOT NULL DEFAULT 'PENDING'")
                )
            if "reviewer_notes" not in existing_cols:
                conn.execute(
                    text("ALTER TABLE prediction_history ADD COLUMN reviewer_notes TEXT")
                )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
