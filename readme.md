# Math Solver with RAG System

This project implements a Math Solver API with a Retrieval-Augmented Generation (RAG) system. It uses FastAPI for the web framework, LangChain for the RAG implementation, and integrates with OpenAI's language models.

## Features

- Solve math problems using traditional methods
- Solve math problems using a RAG system based on uploaded PDF content
- Upload and process PDF documents for the RAG system
- Retrieve similar questions from previously solved problems

## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/AdebisiJoe/fastapi-math-solver.git
   cd fastapi-math-solver
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the FastAPI server:
   ```
   uvicorn main:app --reload
   ```

2. The API will be available at `http://127.0.0.1:8000`

3. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`

## API Endpoints

- `/solve`: Solve a math problem using traditional methods
- `/solve-rag`: Solve a math problem using the RAG system
- `/similar`: Find similar questions to a given problem
- `/upload_pdf`: Upload a PDF document to be used by the RAG system
- `/view_pdf_content`: View the processed content of the uploaded PDF

## Viewing API Documentation

To explore and interact with the API, visit the Swagger UI documentation:

```
http://127.0.0.1:8000/docs
```

This interactive interface allows you to:
- See all available endpoints
- Test API calls directly from the browser
- View request and response schemas

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.