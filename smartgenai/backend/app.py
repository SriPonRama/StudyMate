from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from backend.utils import (
    extract_text_from_pdf,
    calculate_top_terms,
    create_bar_chart,
    create_pie_chart,
    get_file_extension,
    generate_unique_id,
    get_file_path,
    get_static_image_path,
    get_static_image_as_base64,
    chunk_text,
    generate_mind_map,
    generate_power_hour_content,
)
from backend.hf_utils import hf_infer
from backend.bm25_retriever import BM25
from backend.db import engine, create_db_and_tables
from sqlmodel import Session, select
from backend.models import Document, Chunk, Quiz, StudyPlanBlock
from datetime import date

app = FastAPI()

# Create tables on startup
@app.on_event("startup")
def startup_event():
    create_db_and_tables()

# --- Document Upload and Processing ---
@app.post("/docs/upload", response_model=Dict)
async def upload_document(file: UploadFile = File(...)):
    if file.size > 25 * 1024 * 1024:  # 25MB limit
        raise HTTPException(status_code=400, detail="File size exceeds the limit")

    file_extension = get_file_extension(file.filename)
    if file_extension != ".pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are allowed.")

    doc_id = generate_unique_id(file.filename + str(datetime.now())) # use filename and timestamp
    file_path = get_file_path(doc_id, file.filename)

    try:
        with open(file_path, "wb") as f:
            while chunk := file.file.read(1024 * 1024):  # Read in 1MB chunks
                f.write(chunk)
        file.file.close()  # Close the file explicitly

        text = extract_text_from_pdf(file_path)
        top_terms = calculate_top_terms(text)
        top_terms_json = json.dumps(top_terms) # Store as JSON

        # Save document metadata to the database
        with Session(engine) as session:
            db_document = Document(doc_id=doc_id, file_name=file.filename, top_terms=top_terms_json)
            session.add(db_document)
            session.commit()
            session.refresh(db_document)

        return {"doc_id": doc_id, "file_name": file.filename, "top_terms": top_terms}

    except Exception as e:
        print(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# --- Visualization Endpoints ---
@app.get("/viz/{doc_id}/chart/bar")
async def get_bar_chart(doc_id: str):
    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        results = session.exec(statement)
        doc = results.one_or_none()

        if not doc or not doc.top_terms:
            raise HTTPException(status_code=404, detail="Document not found or top terms not available.")

        try:
            top_terms = json.loads(doc.top_terms)
            image_data = create_bar_chart(top_terms)
            if image_data:
                return FileResponse(path=f"/tmp/temp_bar_chart.png", filename="bar_chart.png", media_type="image/png")
            else:
                raise HTTPException(status_code=500, detail="Could not create bar chart.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid top terms data.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating bar chart: {e}")

@app.get("/viz/{doc_id}/chart/pie")
async def get_pie_chart(doc_id: str):
    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        results = session.exec(statement)
        doc = results.one_or_none()

        if not doc or not doc.top_terms:
            raise HTTPException(status_code=404, detail="Document not found or top terms not available.")
        try:
            top_terms = json.loads(doc.top_terms)
            image_data = create_pie_chart(top_terms)
            if image_data:
                return FileResponse(path=f"/tmp/temp_pie_chart.png", filename="pie_chart.png", media_type="image/png")
            else:
                raise HTTPException(status_code=500, detail="Could not create pie chart.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid top terms data.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating pie chart: {e}")

# --- Indexing ---
@app.post("/index/{doc_id}")
async def build_index(doc_id: str):
    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        results = session.exec(statement)
        doc = results.one_or_none()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = get_file_path(doc_id, doc.file_name)
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        # Build BM25 index
        bm25_index_data = build_bm25_index(chunks)

        # Save chunks and embeddings to the database
        for chunk_id, chunk_text in chunks:
            db_chunk = Chunk(doc_id=doc_id, chunk_id=chunk_id, text=chunk_text)
            session.add(db_chunk)
        session.commit()

        return {"message": "Index build started."}

@app.get("/index/{doc_id}/status")
async def get_index_status(doc_id: str):
    with Session(engine) as session:
        statement = select(Chunk).where(Chunk.doc_id == doc_id)
        chunks = session.exec(statement).all()
        if chunks:
             return {"status": "ready"}
        else:
            return {"status": "building"} # Assuming this is always fast enough to be ready

# --- Question Answering ---
@app.post("/qa/ask", response_model=Dict)
async def ask_question(data: Dict):
    doc_id = data.get("doc_id")
    question = data.get("question")

    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="Missing doc_id or question")

    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        doc = session.exec(statement).one_or_none()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = get_file_path(doc_id, doc.file_name)
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        # Build BM25 index (in memory, as it's assumed this is available)
        bm25_index = BM25(corpus=[chunk[1] for chunk in chunks])

        # Get BM25 scores
        top_n = 3 # Number of top chunks to return
        top_results = bm25_index.get_top_n(question, top_n=top_n)

        # Prepare results
        results = []
        for chunk_id, score in top_results:
            chunk_text = next((chunk_text for ch_id, chunk_text in chunks if ch_id == chunk_id), "") # Find text

            results.append({"chunk_id": chunk_id, "score": score, "text": chunk_text})

        # Get extractive answer (BM25)
        extractive_answer = ""
        if results:
             extractive_answer = "\n".join([result["text"] for result in results])

        # Enhance with HF if API key is available
        hf_answer = None
        try:
            hf_answer = hf_infer(f"Context: {extractive_answer}\nQuestion: {question}\nAnswer:", model="google/flan-t5-small")
        except Exception as e:
            print(f"Error during HF inference: {e}")

        answer = hf_answer if hf_answer else extractive_answer

        return {"answer": answer, "spans": results}

# --- Study Plan Generation ---
@app.post("/plan/create", response_model=List[Dict])
async def create_study_plan(data: Dict):
    subjects = data.get("subjects", [])
    exam_date_str = data.get("exam_date")
    hours_per_day = data.get("hours_per_day", 2)
    mode = data.get("mode", "focused")
    mood = data.get("mood", "motivated")

    if not subjects or not exam_date_str:
        raise HTTPException(status_code=400, detail="Missing subjects or exam_date")

    try:
        exam_date = date.fromisoformat(exam_date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid exam_date format.  Use YYYY-MM-DD.")

    # Placeholder for plan generation (Replace with your logic)
    plan_blocks = []
    for subject in subjects:
        plan_blocks.append({
            "block_type": "lecture",
            "subject": subject,
            "date": exam_date,
            "hours": hours_per_day,
            "description": f"Review {subject} concepts and practice problems. (Placeholder - Implement real generation here.)",
        })

    with Session(engine) as session:
        plan_id = generate_unique_id(str(datetime.now()) + str(subjects) + str(exam_date) + str(hours_per_day) + str(mode) + str(mood))
        for block in plan_blocks:
             db_block = StudyPlanBlock(plan_id = plan_id, block_type = block["block_type"], subject = block["subject"], date = block["date"], hours = block["hours"], description = block["description"])
             session.add(db_block)
        session.commit()

    return plan_blocks

# --- Quiz Generation ---
@app.post("/quiz/generate", response_model=Dict)
async def generate_quiz(data: Dict):
    doc_id = data.get("doc_id")
    count = data.get("count", 5)

    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing doc_id")

    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        doc = session.exec(statement).one_or_none()

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = get_file_path(doc_id, doc.file_name)
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        # Placeholder:  Replace with actual quiz generation logic
        questions = [
            {"question": f"Question {i+1}?", "options": ["A", "B", "C", "D"]}
            for i in range(count)
        ]
        answers = [0] * count # Placeholder - indicate which option is correct

        quiz_id = generate_unique_id(f"quiz_{doc_id}_{datetime.now()}")

        db_quiz = Quiz(quiz_id=quiz_id, doc_id=doc_id, questions=json.dumps(questions), answers=json.dumps(answers))
        session.add(db_quiz)
        session.commit()
        session.refresh(db_quiz)

        return {"quiz_id": quiz_id}

# --- Get Quiz Questions ---
@app.get("/quiz/{quiz_id}", response_model=List[Dict])
async def get_quiz_questions(quiz_id: str):
    with Session(engine) as session:
        statement = select(Quiz).where(Quiz.quiz_id == quiz_id)
        quiz = session.exec(statement).one_or_none()
        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")
        try:
            questions = json.loads(quiz.questions)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid quiz data.")

        return questions

# --- Answer Quiz Question ---
@app.post("/quiz/answer", response_model=Dict)
async def answer_quiz_question(data: Dict):
    quiz_id = data.get("quiz_id")
    question_index = data.get("question_index")
    answer = data.get("answer")

    if not quiz_id or question_index is None or answer is None:
        raise HTTPException(status_code=400, detail="Missing quiz_id, question_index, or answer")

    with Session(engine) as session:
        statement = select(Quiz).where(Quiz.quiz_id == quiz_id)
        quiz = session.exec(statement).one_or_none()

        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found")

        try:
            answers = json.loads(quiz.answers) # load correct answer
            # backend/app.py (Continued)
            correct_answer_index = answers[question_index]
            correct = False
            explanation = "Placeholder explanation"
            asset = None
            if isinstance(correct_answer_index, int):
                 # Assume options are indexed 0, 1, 2, 3 ...
                 if answer == quiz.questions[question_index]['options'][correct_answer_index]:
                    correct = True
                    explanation = "Correct!" # Improve the explanation!
            elif isinstance(correct_answer_index, str): # Text-based answer
                if answer.strip().lower() == correct_answer_index.strip().lower():
                    correct = True
                    explanation = "Correct!"

            if correct:
                # Example:  Provide a congratulatory image
                asset = get_static_image_path("congrats.png") # or provide a dynamic image path
            else:
                asset = get_static_image_path("try_again.png")
            return {"correct": correct, "explanation": explanation, "asset": asset}

        except (json.JSONDecodeError, IndexError, TypeError) as e:
            raise HTTPException(status_code=500, detail=f"Error processing quiz answer: {e}")

# --- Recall Map Generation ---
@app.get("/recall/{doc_id}/map.png")
async def get_recall_map(doc_id: str):
    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        doc = session.exec(statement).one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = get_file_path(doc_id, doc.file_name)
        text = extract_text_from_pdf(file_path)

        image_data = generate_mind_map(text, doc_id)
        if image_data:
            return FileResponse(path=f"/tmp/temp_mind_map.png", filename="mind_map.png", media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail="Could not generate mind map.")

# --- Power Hour Generation ---
@app.post("/power/{doc_id}")
async def generate_power_hour(doc_id: str):
    with Session(engine) as session:
        statement = select(Document).where(Document.doc_id == doc_id)
        doc = session.exec(statement).one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = get_file_path(doc_id, doc.file_name)
        text = extract_text_from_pdf(file_path)
        power_hour_content = generate_power_hour_content(text)

        # In a real implementation, you'd generate PDFs here and return URLs
        # Example:
        # notes_pdf_path = generate_notes_pdf(text, doc_id)
        # formulas_pdf_path = generate_formulas_pdf(text, doc_id)
        # top_10_terms_pdf_path = generate_top_10_terms_pdf(text, doc_id)

        # Placeholder: Return dummy URLs
        return {
            "notes_url": f"/static/power_hours/{doc_id}/notes.pdf",
            "formulas_url": f"/static/power_hours/{doc_id}/formulas.pdf",
            "top_10_terms_url": f"/static/power_hours/{doc_id}/top_10.pdf",
        }