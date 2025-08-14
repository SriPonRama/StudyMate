
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.db import create_db_and_tables, engine
from sqlmodel import Session
from backend.models import Document, Chunk

@pytest.fixture(scope="module")
def test_app():
    create_db_and_tables()
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def sample_pdf_path():
    # Create a dummy PDF file for testing (replace with a real PDF path if you want)
    with open("test.pdf", "w") as f:
        f.write("%PDF-1.7\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 0>>\nstream\nBT/F1 12 Tf 100 700 Td (Hello, world!) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000061 00000 n\n0000000112 00000 n\n0000000175 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n228\n%%EOF")
    yield "test.pdf"
    import os
    os.remove("test.pdf")

@pytest.fixture(scope="module")
def sample_doc_id(test_app, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        response = test_app.post("/docs/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    return data["doc_id"]

def test_upload_document(test_app):
    with open("test.pdf", "w") as f:
        f.write("%PDF-1.7\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 0>>\nstream\nBT/F1 12 Tf 100 700 Td (Hello, world!) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000061 00000 n\n0000000112 00000 n\n0000000175 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n228\n%%EOF")
    with open("test.pdf", "rb") as f:
        files = {"file": ("test.pdf", f, "application/pdf")}
        response = test_app.post("/docs/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    import os
    os.remove("test.pdf")

def test_build_index(test_app, sample_doc_id):
    response = test_app.post(f"/index/{sample_doc_id}")
    assert response.status_code == 200

def test_ask_question(test_app, sample_doc_id):
    # First build the index
    test_app.post(f"/index/{sample_doc_id}") # Build the index
    # Then, ask a question
    response = test_app.post("/qa/ask", json={"doc_id": sample_doc_id, "question": "What does this document say?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "spans" in data

def test_create_study_plan(test_app):
    response = test_app.post("/plan/create", json={
        "subjects": ["Math", "Physics"],
        "exam_date": "2024-07-20",
        "hours_per_day": 2,
        "mode": "focused",
        "mood": "motivated"
    })
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0