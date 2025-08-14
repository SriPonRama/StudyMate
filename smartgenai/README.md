
# Exam Prep App - Backend

This is the backend for a gamified, offline-first exam preparation app built with FastAPI and designed for enhanced functionality through optional integration with the Hugging Face Inference API. The application prioritizes offline usability, ensuring core features remain accessible even without an internet connection.  Online features are gracefully degraded when the Hugging Face API key is not provided or the API is unavailable.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [Setup](#setup)
    *   [1. Cloning the Repository](#1-cloning-the-repository)
    *   [2. Creating and Activating a Virtual Environment](#2-creating-and-activating-a-virtual-environment)
    *   [3. Installing Dependencies](#3-installing-dependencies)
    *   [4. Setting Environment Variables](#4-setting-environment-variables)
    *   [5. VS Code Configuration](#5-vs-code-configuration)
    *   [6. Running the Application](#6-running-the-application)
*   [Usage](#usage)
    *   [1. Uploading a PDF Document (/docs/upload)](#1-uploading-a-pdf-document-docsupload)
    *   [2. Building an Index (/index/{doc\_id})](#2-building-an-index-indexdoc_id)
    *   [3. Asking a Question (/qa/ask)](#3-asking-a-question-qaask)
    *   [4. Generating a Study Plan (/plan/create)](#4-generating-a-study-plan-plancreate)
    *   [5. Generating a Quiz (/quiz/generate)](#5-generating-a-quiz-quizgenerate)
    *   [6. Getting Quiz Questions (/quiz/{quiz\_id})](#6-getting-quiz-questions-quizquiz_id)
    *   [7. Answering a Quiz Question (/quiz/answer)](#7-answering-a-quiz-question-quizanswer)
    *   [8. Generating a Recall Map (/recall/{doc\_id}/map.png)](#8-generating-a-recall-map-recalldoc_idmap.png)
    *   [9. Generating Power Hour PDFs (/power/{doc\_id})](#9-generating-power-hour-pdfs-powerdoc_id)
*   [API Documentation](#api-documentation)
*   [Frontend Integration (Example JavaScript Snippets)](#frontend-integration-example-javascript-snippets)
    *   [1. Upload Document](#1-upload-document)
    *   [2. Build Index](#2-build-index)
    *   [3. Ask Question](#3-ask-question)
    *   [4. Create Study Plan](#4-create-study-plan)
    *   [5. Generate Quiz](#5-generate-quiz)
    *   [6. Get Quiz Questions](#6-get-quiz-questions)
    *   [7. Answer Quiz Question](#7-answer-quiz-question)
    *   [8. Get Recall Map](#8-get-recall-map)
    *   [9. Generate Power Hour PDFs](#9-generate-power-hour-pdfs)
*   [Offline-First Behavior](#offline-first-behavior)
*   [Security & Rate Limiting](#security--rate-limiting)
*   [Troubleshooting](#troubleshooting)
*   [Docker (Optional)](#docker-optional)
    *   [1. Creating a Dockerfile](#1-creating-a-dockerfile)
    *   [2. Building the Docker Image](#2-building-the-docker-image)
    *   [3. Running the Docker Container](#3-running-the-docker-container)
*   [Development Script (Optional)](#development-script-optional)
    *   [1. Creating a `dev.sh` Script](#1-creating-a-devsh-script)
    *   [2. Making the Script Executable](#2-making-the-script-executable)
    *   [3. Running the Script](#3-running-the-script)
*   [Project Structure](#project-structure)

## Prerequisites

*   **Python 3.11+:** Ensure you have a compatible version of Python installed. You can check your Python version by running `python --version` or `python3 --version` in your terminal.
*   **VS Code:**  Visual Studio Code (VS Code) is recommended for development, offering features like code completion, debugging, and integrated terminal support.  Make sure you have the latest version.

## Setup

Follow these steps to set up and run the backend application:

### 1. Cloning the Repository

Use `git clone` to download the project files from the repository.

```bash
git clone <your_repository_url>
cd <your_repository_name>