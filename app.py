import torch
from huggingface_hub import login

from llama_index.core import VectorStoreIndex,ServiceContext,SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding

import streamlit as st

st.set_page_config(page_title="Building RAGS with LLAMA", layout="wide")

st.markdown("""
## Laws of Cricket 2017- 3rd Edition 2022: Get instant insights from the laws pdf file

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Metas's Generative AI model Llama-3-8b. It processes uploaded PDF document comprising all the laws of cricket, creates a searchable vector store, and generates accurate answers to user queries. 

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your Hugging Face Token**: You'll need to generate a hugging face token to access the Meta's open-source LLama models. Obtain your token and get permission from Meta

3. **Ask a Question**: Ask any question related to the cricket laws, this bot will be Assistant to the umpires.
""")

with st.expander("Setting the LLM"):
    with st.form("setting"):
        row_1 = st.columns(3)
        
        with row_1[0]:
            token = st.text_input("Hugging Face Token", type="password")

        with row_1[1]:
            llm_model = st.text_input("LLM model", value="meta-llama/Meta-Llama-3-8B-Instruct")

        with row_1[2]:
            instruct_embeddings = st.text_input("Instruct Embeddings", value="sentence-transformers/all-mpnet-base-v2")

        row_2 = st.columns(1)
        
        
        create_chatbot = st.form_submit_button("Create chatbot")

if token!="":

    login(token)
    #Initialize Message History
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about Streamlit's open-source Python library!"}
        ]

    system_prompt="""

    You are a Q&A assistant. Your goal is to answer questions as
    accurately as possible based on the instructions and context provided.

    """

    #Default prompt supported by llama2
    query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")


    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.0, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        # loading model in 8bit for reducing memory
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
    )

    embed_model= LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
            reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = reader.load_data()
            service_context = ServiceContext.from_defaults(chunk_size=1024,llm=llm,embed_model=embed_model)
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index

    index = load_data()

    chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history

else:
    st.write("No hugging face token provided")
