from sqlmodel import SQLModel, create_engine
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./exam_prep.db")

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)