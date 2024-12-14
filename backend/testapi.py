import os
import json
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

index_name = "case-study-index"

def load_and_process_data(filepath):
    cases_df = pd.read_csv(filepath)
    cases_df["combined_text"] = cases_df.apply(
        lambda row: (
            f"Case Name: {row['Case Name']}. "
            f"Semester: {row['Semester']}. "
            f"Company Name: {row['Company Name']}. "
            f"Description: {row['Detailed Case Description']}. "
            f"Outcome: {row['Outcome']}. "
            f"Tech Stack: {row['Tech Stack']}. "
            f"KPI: {row['KPIs']}. "
            f"Quoted Price: {row['Quoted Price']}."
        ),
        axis=1
    )
    return cases_df

def setup_pinecone_index(cases_df):
    embeddings = OpenAIEmbeddings()
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(index, embeddings, "text")
    texts = cases_df["combined_text"].tolist()
    metadatas = cases_df.to_dict('records')
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    return vectorstore

def generate_email(company_info, retriever):
    company_text = f"""
    Company Name: {company_info.get('Company Name', '')}
    Industry: {company_info.get('Industry', '')}
    Technologies: {company_info.get('Technologies', '')}
    Keywords: {company_info.get('Keywords', '')}
    """

    retrieved_cases = retriever.invoke(company_text)

    retrieved_cases_text = "\n".join([
        f"Case Name: {doc.metadata['Case Name']}, Outcome: {doc.metadata['Outcome']}, Quoted Price: {doc.metadata['Quoted Price']}"
        for doc in retrieved_cases
    ])

    prompt_template = """
    [Content from earlier code, unchanged for brevity.]
    """
    prompt = PromptTemplate(input_variables=["company_info", "retrieved_cases"], template=prompt_template)
    final_prompt = prompt.format(company_info=company_text, retrieved_cases=retrieved_cases_text)

    # Generate email using LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    messages = [{"role": "system", "content": "You are a professional email writer."},
                {"role": "user", "content": final_prompt}]
    response = llm.invoke(messages)
    return response.content

# Lambda Handler
def lambda_handler(event, context):
    try:
        # Parse input data from API Gateway event
        body = json.loads(event.get("body", "{}"))
        company_info = {
            "Company Name": body.get("Company Name", ""),
            "Industry": body.get("Industry", ""),
            "Technologies": body.get("Technologies", ""),
            "Keywords": body.get("Keywords", "")
        }

        # Load case study data and set up Pinecone
        cases_df = load_and_process_data("dummy_data.csv")  # Replace with the appropriate data source
        vectorstore = setup_pinecone_index(cases_df)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Generate email
        email_content = generate_email(company_info, retriever)

        # Return successful response
        return {
            "statusCode": 200,
            "body": json.dumps({"email_content": email_content}),
            "headers": {"Content-Type": "application/json"}
        }
    except Exception as e:
        # Handle errors gracefully
        print("Error:", e)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
            "headers": {"Content-Type": "application/json"}
        }
