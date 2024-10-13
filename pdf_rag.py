from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI  , OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class PDFRAGSystem:
    def __init__(self, persist_directory="./pdf_rag_db"):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, )
        self.vector_store = Chroma(embedding_function=self.embeddings, persist_directory=persist_directory)
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a math tutor using the NEW GENERAL MATHEMATICS for Junior Secondary Schools JSS Teacher's Guide. 
            Use the following context to help answer the question. If the context isn't directly relevant, use your general knowledge.

            Context:
            {context}

            Question:
            {question}

            Provide a step-by-step solution and suggest related topics from the guide for further study:
            """
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt_template}
        )

    def process_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Use only the first 20 pages
        pages = pages[:20]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(pages)
        
        self.vector_store.add_documents(texts)

    def answer_question(self, question):
        result = self.qa_chain({"query": question})
        return result['result']

    def get_similar_content(self, question):
        results = self.vector_store.similarity_search(question, k=5)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]