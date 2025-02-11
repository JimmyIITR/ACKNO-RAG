import os
import time
import pandas as pd
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Initialize Chroma client and persistent directory
CHROMA_CLIENT = chromadb.Client()
persist_directory = "chroma_index"

def get_embeddings():
    """Initialize Hugging Face embeddings"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_deepseek_r1_client():
    """Initialize DeepSeek-R1 client"""
    return InferenceClient(
        provider="together",
        api_key=os.getenv("HF_TOKEN")
    )

def main():
    # Load the CSV file
    input_file = "../data/testData100.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    data_df = pd.read_csv(input_file)
    required_columns = ["factcheck_analysis_link", "statement"]
    if not all(col in data_df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    urls = data_df["factcheck_analysis_link"].dropna().tolist()
    print(f"Processing {len(urls)} URLs...")

    # Step 1: Process URLs and create vector database
    loader = UnstructuredURLLoader(urls=urls)
    print("Loading data from URLs...")
    data = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    print("Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    # Create embeddings and store in Chroma
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=CHROMA_CLIENT,
        persist_directory=persist_directory
    )
    print("Building embedding vectors...")
    time.sleep(2)
    vectorstore.persist()
    print("Chroma index saved")

    # Step 2: Process queries
    print("Processing queries...")
    data_df["Result"] = None
    
    client = get_deepseek_r1_client()
    template = """Analyze the context and determine if the statement is True/False. Answer only with "True", "False", or "Insufficient data".

    Context: {context}
    Statement: {question}
    Answer:"""

    for index, row in data_df.iterrows():
        query = row["statement"]
        if pd.isna(query) or not query.strip():
            continue
            
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        docs = retriever.invoke(query)
        
        if not docs:
            data_df.at[index, "Result"] = "Insufficient data"
            continue
            
        context = "\n\n".join([d.page_content for d in docs])
        
        # Format prompt
        prompt = template.format(context=context, question=query)
        
        try:
            response = client.chat.completions.create(
                model="deepseek-ai/DeepSeek-R1",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            result = response.choices[0].message.content.strip()
            # Post-process response
            if "true" in result.lower():
                data_df.at[index, "Result"] = "True"
            elif "false" in result.lower():
                data_df.at[index, "Result"] = "False"
            else:
                data_df.at[index, "Result"] = "Insufficient data"
        except Exception as e:
            print(f"Error processing query: {e}")
            data_df.at[index, "Result"] = "Error"

    # Save results
    output_file = "../results/test_with_results.csv"
    data_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()