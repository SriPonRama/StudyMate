

from sqlmodel import SQLModel, Field
from typing import Optional, List
from datetime import date

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: str
    file_name: str
    top_terms: str  # Stored as a JSON string, e.g., '{"term1": 0.5, "term2": 0.3}'

class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: str
    chunk_id: str
    text: str
    embedding: Optional[str] = Field(default=None) # JSON encoded list of floats

class Quiz(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    quiz_id: str
    doc_id: str
    questions: str  # JSON string of questions
    answers: str  # JSON string of answers (correct indices or answers)

class StudyPlanBlock(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    plan_id: str # Group blocks together in a "plan"
    block_type: str # e.g., "lecture", "practice", "review"
    subject: str
    date: date
    hours: float
    description: str # Details about the block's activities