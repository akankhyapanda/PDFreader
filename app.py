import streamlit as st
import os
import tempfile
import textract
import fitz
from PIL import Image
import io
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

os.environ['OPENAI_API_KEY'] = ' '

# Define a class for session state
class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# Function to get session state
def get_session_state():
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()
    return st.session_state.session_state

# Function to get or create conversation history in session state
def get_or_create_conversation_history():
    session_state = get_session_state()
    if not hasattr(session_state, 'conversation_history'):
        session_state.conversation_history = []
    return session_state.conversation_history

#get both pdf or doc text
def get_pdf_text(file_paths):
    text = ""
    for file in file_paths:
        file_extension = os.path.splitext(file)[-1].lower()

        if file_extension == '.pdf':
            try:
                # Extract text from PDF
                text += textract.process(file).decode('utf-8')
            except Exception as e:
                print(f"Error extracting text from PDF ({file}): {e}")
        elif file_extension in ('.doc', '.docx'):
            try:
                # Extract text from DOC or DOCX
                text += textract.process(file).decode('utf-8')
            except Exception as e:
                print(f"Error extracting text from DOCX ({file}): {e}")

    return text
 
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks
 
 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    


llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

def main():
    st.sidebar.title("CHATAI")
    st.sidebar.markdown("---") 

    # Display the options menu
    choice = st.sidebar.radio("Menu", ['New Chat', 'Chat History', 'Usage Statistics'])

    if choice == 'New Chat':
        col2, col3 = st.columns([0.4, 0.6], gap="large")
        temp_file_path = None

        with col2:
            st.title("New Chat")

            st.write("Upload a PDF or DOCX file:")
            uploaded_files = st.file_uploader("", accept_multiple_files=True, type=["pdf", "docx"], key="pdf_docx_files")
 
            if uploaded_files is not None:
                st.success("Files uploaded successfully!")

            if uploaded_files:
                temp_file_paths = []

                for uploaded_file in uploaded_files:
                    # Save the uploaded file to a temporary location
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        temp_file_path = tempfile.mktemp(suffix=uploaded_file.name)

                        with open(temp_file_path, "wb") as temp_file:
                            temp_file.write(uploaded_file.read())

                        temp_file_paths.append(temp_file_path)
        
        
        with col3:
            st.title("PDF Preview")
            
            if uploaded_files is not None:
                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"**{uploaded_file.name}**")
                    
                    
                    if uploaded_file.type == 'application/pdf':
                        document = fitz.open(temp_file_paths[i])

                        page_number = st.session_state.get(f'page_number_{uploaded_file.name}', 0)  

                        if st.button(f"Previous Page {uploaded_file.name}"):
                            page_number -= 1  # Decrement the page number
                            if page_number < 0:
                                page_number = 0  
                            st.session_state[f'page_number_{uploaded_file.name}'] = page_number  

                        if st.button(f"Next Page {uploaded_file.name}"):
                            page_number += 1  # Increment the page number
                            st.session_state[f'page_number_{uploaded_file.name}'] = page_number  
                        if page_number < document.page_count:
                            page = document.load_page(page_number)
                            image_bytes = page.get_pixmap().tobytes()
                
                            img = Image.open(io.BytesIO(image_bytes))
                
                            st.image(img, caption=f"Page {page_number + 1} of {uploaded_file.name}", width=500)

                        document.close()
                        
                    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                        # Extract text content from Word document
                        raw_text = get_pdf_text([temp_file_paths[i]])
                        st.text_area(f"Document Content: {uploaded_file.name}", raw_text, height=300)
                        
                        
        # Display conversation
        new_prompt = st.text_input("Enter your message:")
        if new_prompt:

            conversation_history = get_or_create_conversation_history()
            conversation_history.append({"role": "user", "content": new_prompt})

            raw_text = get_pdf_text(temp_file_paths)
            # get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # create vector store
            vectorstore = get_vectorstore(text_chunks)

            # Create VectorStoreToolkit and agent executor
            vectorstore_info = VectorStoreInfo(
                name="Pdf/doc",
                description="pdf/doc file ",
                vectorstore=vectorstore
            )

            toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

            agent_executor = create_vectorstore_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True
            )

            response = agent_executor.run(new_prompt)
            conversation_history.append({"role": "model", "content": response})
        
        # Display entire conversation history
        conversation_history = get_or_create_conversation_history()
        for chat in conversation_history:
            st.text(f"{chat['role'].capitalize()}: {chat['content']}")

    elif choice == 'Chat History':
        st.title("Chat History")

    elif choice == 'Usage Statistics':
        col6, col7 = st.columns([0.7, 0.3], gap="small")

        with col6:
            st.title("files:")

        with col7:
            st.title("Prompts:")


if __name__ == "__main__":
    main()
