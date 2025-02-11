import os
import time
import pandas as pd
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
import chromadb

# Load environment variables
load_dotenv()

# Initialize Chroma client and persistent directory
CHROMA_CLIENT = chromadb.Client()
persist_directory = "chroma_index"

def get_openai_embeddings():
    """Function to initialize OpenAI embeddings."""
    return OpenAIEmbeddings(model="text-embedding-ada-002")

def main():
    # Load the CSV file
    input_file = "../data/testData100.csv"  # Use your generated CSV here
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    data_df = pd.read_csv(input_file)
    if "factcheck_analysis_link" not in data_df.columns or "statement" not in data_df.columns:
        print("Error: CSV file must contain columns named 'factcheck_analysis_link' and 'statement'.")
        return

    urls = data_df["factcheck_analysis_link"].dropna().tolist()
    print(f"Processing {len(urls)} URLs...")

    # Step 1: Process all URLs and create a vector database
    loader = UnstructuredURLLoader(urls=urls)
    print("Loading data from URLs...")
    data = loader.load()

    # Split data into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    print("Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    # Create embeddings and save to Chroma
    embeddings = get_openai_embeddings()
    vectorstore_chroma = Chroma.from_documents(
        docs,
        embedding=embeddings,
        client=CHROMA_CLIENT,
        persist_directory=persist_directory
    )
    print("Building embedding vectors...")
    time.sleep(2)

    print("Saving Chroma index...")

    # Step 2: Process each query and retrieve results
    print("Processing queries and retrieving results...")
    data_df["Result"] = None  # Add a column for results
    for index, row in data_df.iterrows():
        query = row["statement"]
        if query:
            embeddings = get_openai_embeddings()
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
                client=CHROMA_CLIENT
            )

            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.80, "k": 1}
            )
            results = retriever.invoke(query)

            if not results:
                data_df.at[index, "Result"] = "Insufficient data"
            else:
                context = "\n\n".join([doc.page_content for doc in results])

                template = """Use the following pieces of context to check if the given query is factually correct or not.
                Clearly state if the given query is factually correct according to the context (yes / no / insufficient data).
                Use three sentences maximum and keep the answer concise. 
                Example 1:
                Context: In his State of the Union address on Jan. 27, 2010, President Barack Obama unveiled a proposal to freeze discretionary spending for three years to help dig the country out of a "massive fiscal hole."
                But Obama included some caveats. He said the freeze would exclude spending on national security as well as entitlement programs such as Medicare, Medicaid and Social Security. And the freeze won't take effect until next year, "when the economy is stronger," he said.
                "Understand," Obama said, "if we don't take meaningful steps to rein in our debt, it could damage our markets, increase the cost of borrowing, and jeopardize our recovery -– all of which would have an even worse effect on our job growth and family incomes."
                That kind of belt-tightening talk makes for good sound bites, Republican leaders said afterward, but the freeze described by Obama, while a small step in the right direction, amounts to a tiny sliver of the country's massive (and growing) debt.
                "President Obama's proposed 'spending freeze' will only reduce the $42 trillion in government spending proposed between 2011 and 2020 by little more than one-half of 1 percent," House Republican Leader John Boehner wrote in a live blog during Obama's speech.
                But first, are Boehner's numbers themselves accurate?
                A report released this week from the nonpartisan Congressional Budget Office projected the federal government will spend $42.9 trillion over the next 10 years. The Obama administration itself says the freeze on nondefense discretionary spending would save $250 billion over 10 years. That number, however, is a bit more speculative than the CBO's overall estimate.
                Speaking before the Senate Budget Committee on Jan. 28, 2010, Douglas W. Elmendorf, director of the Congressional Budget Office, said his office did not have enough details from the White House about which categories of spending would be exempt from Obama's pledge. Those details will be revealed when the White House presents its proposed 2011 budget next week. But Elmendorf said perhaps the most crucial factor in determining the long-term effect of the freeze is what happens in the years after the freeze. If spending immediately goes back to the levels they would have been at had they been rising all along, "then the savings are just in those three years, and they're small," Elmendorf said.
                For example, Elmendorf said, a freeze on all discretionary appropriations would only save about $10 billion in fiscal year 2011.
                But, he said, if you essentially reset spending levels and only increase at the rate of inflation after the freeze, "then you can achieve significant savings over the remaining years." That's the route White House officials have said they want to go.
                So that $250 billion figure cited by the White House seems in line with other estimates.
                And by that measure, Boehner's attempt to put Obama's proposal into context is accurate. A freeze that would cut spending by $250 billion over 10 years amounts to a little more than one half of 1 percent of all government spending over that same period.
                But he also relied on math that made Obama's freeze look especially tiny because the other number he cited -- $42 trillion in overall spending -- includes mandatory entitlement programs such as Social Security, Medicare and Medicaid as well as discretionary nondefense and defense programs. If you remove those items, Obama's freeze accounts for a slightly larger share -- about 4 percent over 10 years.
                The White House, however, puts a more optimistic spin on this statistic saying that by the middle of the decade, nonsecurity discretionary spending would reach its lowest percentage of gross domestic product in 50 years.
                Still the big-picture point made by Boehner and other Republicans is that Obama's proposed three-year spending freeze, which would apply only to about 17 percent of overall spending, does little to address the "massive fiscal hole" he described.
                "In sum, the outlook for the federal budget is bleak," Elmendorf said.
                So how much of a dent would Obama's freeze make?
                "As a share of the total deficit problem, it is a small step," Elmendorf said.
                So Obama positioned the spending freeze as a way to rein in the country's mounting debt, and we think it's fair for Boehner to put those spending cuts in the context of overall spending. We rate Boehner's statement True.
                
                Question: President Obama's proposed 'spending freeze' will only reduce the $42 trillion in government spending proposed between 2011 and 2020 by little more than one-half of 1 percent.
                
                Answer: True

                Example 2:
                Context: A study published Jan. 1 found that — of 1,313 symptomatic COVID-19 patients, including 862 omicron patients as of Dec. 20 in the Houston Methodist healthcare system — 15% of omicron patients were hospitalized, compared with 43% of delta variant patients and 55% of alpha patients. 
                People who contracted the omicron variant were about half as likely to need hospital care as those infected with delta, according to a report issued Dec. 31 by the UK Health Security Agency. The study included 815 people with omicron who were admitted to hospitals or transferred from emergency departments.
                Omicron was first identified by researchers in South Africa, who reported it to the World Health Organization on Nov. 24. The World Health Organization designated it as a variant of concern two days later. It has spread to at least 110 countries. 
                The first U.S. case was confirmed by the U.S. Centers for Disease Control and Prevention on Dec. 1, days after a Californian returned home from South Africa. 
                On Jan. 4, the CDC estimated that 95.4% of the COVID-19 cases in the United States in the week ending Jan. 1 were the omicron variant and just 4.6% from the delta variant. A month earlier, Omicron’s share was just 0.6%.
                While there is not comprehensive data on omicron hospitalizations, it’s simply not the case that there have been none.
                The World Health Organization told PolitiFact it does not have data on COVID-19 infections that are broken down by variant. While early data from South Africa, the United Kingdom and Denmark suggest a reduced risk of hospitalization for omicron compared to delta, increased transmission due to omicron is expected to lead to more hospitalizations, the organization said. 
                "It is still unclear to what extent the observed reduction in risk of hospitalization can be attributed to immunity from previous infections or vaccination and to what extent omicron may be less virulent," the organization said in an email.
                Omicron hospitalizations were reported before the ad went live.
                
                Question: No one has been hospitalized for"" the omicron variant of COVID-19.
                
                Answer: False

                Conext: {context}
                Question: {question}
                Answer: (True/False)
                """

                QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

                qa_chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(temperature=0.9, max_tokens=500),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(),
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                )

                result = qa_chain.invoke({"query": query})
                data_df.at[index, "Result"] = result["result"]

    # Save the updated CSV file
    output_file = "../results/test_with_results.csv"
    data_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
