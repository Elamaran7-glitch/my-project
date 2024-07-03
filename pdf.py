from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain


pdf_files = ["ponninselvan.pdf", "dhoni.pdf"]

# Load content from each PDF
pages_content = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages_content.extend(loader.load_and_split())
    print(pages_content)



# Create embeddings and FAISS index
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(pages_content, embeddings)
db.save_local("faiss_index")

# Load FAISS index
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id=model_id,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 150}
)

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know. The AI ONLY uses information contained in the "Relevant Information" section and does not hallucinate.

Relevant Information:

{history}

Conversation:
Human: {input}
AI:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)

memory=ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,prompt=prompt
    )

qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_db.as_retriever(),memory=memory,prompt=prompt)
def ask(user_query):
    res = qa_chain.invoke({"query": user_query})
    return res["result"]

while True:
    user_input = input("You: ")  # Get user input

    if user_input.lower() == "exit":  # Check if the user wants to exit
        print("Exiting conversation.")
        break

    # Ask question and get response
    response =ask(user_input)
    print("AI:", response)