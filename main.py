from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from pdf_rag import PDFRAGSystem

# Load environment variables from the .env file
load_dotenv(".env")

# Get the OpenAI API key from the environment variables
#openai_api_key = os.getenv("OPENAI_API_KEY")

#if not openai_api_key:
#    raise ValueError("OpenAI API key is missing. Please set it in the .env file.")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, )

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Initialize Chroma vector store
vector_store = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Create a prompt template for math problem solving
math_prompt = ChatPromptTemplate.from_template(
    "Solve the following math question step by step. Give me the answer using LaTeX: {question}"
)

# Create a chain for math problem solving
math_chain = math_prompt | llm | RunnablePassthrough()

# Initialize PDF RAG System
pdf_rag_system = PDFRAGSystem()

class Question(BaseModel):
    question: str

@app.post("/math/solve")
async def solve_question(question: Question):
    try:
        # Use the chain to generate the solution
        result = math_chain.invoke({"question": question.question})
        
        # Extract the solution string
        if isinstance(result, str):
            solution = result
        elif isinstance(result, dict) and 'text' in result:
            solution = result['text']
        elif hasattr(result, 'content'):
            solution = result.content
        else:
            solution = str(result)
        
        # Create embeddings for the question
        embedding = embeddings.embed_query(question.question)
        
        # Store the question and solution in the Chroma vector store
        vector_store.add_texts(
            [question.question], 
            metadatas=[{"question": question.question, "solution": solution}], 
            embeddings=[embedding]
        )
        
        return {"solution": solution}
    except Exception as e:
        print(f"Error in solve_question: {str(e)}")  # For debugging
        raise HTTPException(status_code=500, detail=f"Failed to solve the question: {str(e)}")

@app.post("/math/solve-rag")
async def solve_question_rag(question: Question):
    try:
        # Use the PDF RAG system to answer the question
        answer = pdf_rag_system.answer_question(question.question)
        
        # Get similar content from the PDF
        similar_content = pdf_rag_system.get_similar_content(question.question)
        
        return {"solution": answer, "similar_content": similar_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to solve the question using RAG: {str(e)}")

@app.post("/math/similar")
async def get_similar_questions(question: Question):
    try:
        # Create embeddings for the input question
        embedding = embeddings.embed_query(question.question)
        
        # Perform a similarity search in Chroma vector store
        results = vector_store.similarity_search_by_vector(embedding, k=5)
        
        # Extract metadata from the search results
        similar_questions = [result.metadata for result in results]
        
        return similar_questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve similar questions: {str(e)}")

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        # Process the PDF
        pdf_rag_system.process_pdf(file.filename)
        
        # Remove the temporary file
        os.remove(file.filename)
        
        return JSONResponse(content={"message": "PDF processed and stored successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.get("/view_vector_rag_content", response_class=HTMLResponse)
async def view_pdf_content():
    try:
        # Check if the vector store exists
        if pdf_rag_system.vector_store is None:
            return JSONResponse(
                status_code=404,
                content={"message": "Vector store not initialized. Please upload a PDF first."}
            )

        # Retrieve a sample of documents from the vector store
        results = pdf_rag_system.vector_store.similarity_search("", k=10)  # Retrieve 10 documents

        if not results:
            return JSONResponse(
                status_code=404,
                content={"message": "No documents found in the vector store. Please upload a PDF first."}
            )

        # Generate HTML table
        html_content = """
        <table border="1">
            <tr>
                <th>Content</th>
                <th>Metadata</th>
                <th>Embedding (first 5 dimensions)</th>
            </tr>
        """

        for doc in results:
            content = doc.page_content
            metadata = doc.metadata

            # Truncate content if it's too long
            if len(content) > 100:
                content = content[:100] + "..."

            # Get embedding for the document
            embedding = pdf_rag_system.embeddings.embed_query(content)
            
            # Truncate embedding to first 5 dimensions
            truncated_embedding = embedding[:5]

            html_content += f"""
            <tr>
                <td>{content}</td>
                <td>{metadata}</td>
                <td>{truncated_embedding}</td>
            </tr>
            """

        html_content += "</table>"

        return HTMLResponse(content=html_content)
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in view_pdf_content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve PDF content: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)