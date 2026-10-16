from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
from app.config import settings

# Configure SQLite vs PostgreSQL parameters
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Applicant input features
    person_age = Column(Integer, nullable=False)
    person_income = Column(Float, nullable=False)
    person_home_ownership = Column(String(50), nullable=False)
    person_emp_length = Column(Float, nullable=True)
    loan_intent = Column(String(50), nullable=False)
    loan_grade = Column(String(10), nullable=False)
    loan_amnt = Column(Float, nullable=False)
    loan_int_rate = Column(Float, nullable=True)
    loan_percent_income = Column(Float, nullable=False)
    cb_person_default_on_file = Column(String(5), nullable=False)
    cb_person_cred_hist_length = Column(Integer, nullable=False)
    
    # Prediction output
    prediction = Column(String(20), nullable=False)
    probability = Column(Float, nullable=False)

def init_db():
    """Initializes database tables according to defined models."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """FastAPI Dependency that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
