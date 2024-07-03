from langchain.document_loaders import PyPDFLoader



loader=PyPDFLoader("ponninselvan.pdf")
pages_content=loader.load_and_split()
print(len(pages_content),pages_content)
print(pages_content)

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embeddings=HuggingFaceEmbeddings()
db=FAISS.from_documents(pages_content,embeddings)

db.save_local("faiss_index")


new_db=FAISS.load_local("faiss_index",embeddings, allow_dangerous_deserialization= True)


query="are they any sports information"
#docs=new_db.similarity_search(query)
#print(docs)

from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
# Specify the Hugging Face model name or model identifier

# os.environ['HUGGINGFACEHUB_API_TOKEN'] ="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"
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceHub(
    huggingfacehub_api_token="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",
    repo_id=model_id,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 150}
)
qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_db.as_retriever())
res = qa_chain.invoke({"query"})
print(res["result"])
#def ask(user_query):
    #res=qa_chain({"query":user_query})
    #return res["resulton