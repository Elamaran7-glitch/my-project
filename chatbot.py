import os
import asyncio
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from IPython.display import display
import chainlit as cl

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"
model_id = "openai-community/gpt2"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
    repo_id=model_id,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 150}
)

template = """You are a programmer. Provide all syntax for all programs based on the input.
{query}
"""

prompt = PromptTemplate(template=template, input_variables=['query'])
conv_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)

@cl.on_chat_start
def main():
    cl.user_session.set("llm_chain", conv_chain)

@cl.on_message
async def on_message(message:str):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(message)
    # Perform post-processing on the received response here
    # res is a dict and the response text is stored under the key "text"
    display(res["text"])  # Display the response

if __name__ == "__main__":
    cl.run()
