import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain.prompts import PromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

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
            f"Tech Stack: {row.get('Tech Stack', 'Unknown')}. "
            # f"Quoted Price: {row.get('Quoted Price', 'Unknown')}."
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

# somewhat temporary function to get the best matched price by looking at the first case in similar cases, could be improved
# noticable flaw for companies that we quote with 0
# def get_best_matched_price(retrieved_cases):
#     similar_cases = [
#         {"Case Name": doc.metadata["Case Name"], "Quoted Price": float(doc.metadata["Quoted Price"])}
#         for doc in retrieved_cases
#         if "Quoted Price" in doc.metadata and doc.metadata["Quoted Price"]
#     ]
    
#     if similar_cases:
#         return similar_cases[0]["Quoted Price"]
#     return None

# generate email for company based on company_info, similar cases, and sourcer's name
def generate_email(company_info, retriever, sourcer_name):
    first_name = company_info.get("First Name", "")
    title = company_info.get("Title", "")
    company_name = company_info.get("Company Name", "")
    industry = company_info.get("Industry", "")
    company_text = f"""
    Company Name: {company_name}
    Industry: {industry}
    Technologies: {company_info.get('Technologies', '')}
    Keywords: {company_info.get('Keywords', '')}
    """

    # Retrieve similar cases
    retrieved_cases = retriever.invoke(company_text)

    # Generate retrieved_cases_text
    if retrieved_cases:
        retrieved_cases_text = "\n".join([
            f"Case Name: {doc.metadata['Case Name']}, Description: {doc.metadata.get('Detailed Case Description', 'No description available')}"
            for doc in retrieved_cases
        ])
    else:
        retrieved_cases_text = "No relevant case studies were found."

    # Define the prompt template
    prompt_template = """
    You are a sourcing analyst for the Harvard Undergraduate Data Analytics Group (HDAG), tasked with drafting a personalized cold outreach email to introduce HDAG's services to potential clients. The goal is to optimize response rates by creating concise, engaging, and tailored emails that highlight HDAG's value proposition.

    **Inputs:**
    - **Company Information:** {company_info}
      - Includes contact details (e.g., first name, last name, title, company, industry), dynamic columns ("brief summary" and "in-depth analysis"), and more.
    - **Relevant Case Studies:** {retrieved_cases}
      - Vectorized dataset of past projects completed by HDAG.

    **Instructions:**
    Write a cold outreach email in three clear paragraphs, designed to maximize response rates, following the structure below:

    ### 1. **Introduction to HDAG + Summary of the Company**
    - Begin with a personalized greeting and introduction:
      > Dear {first_name},  
      > I hope this email finds you well. My name is {sourcer_name}, and I help represent the Harvard Undergraduate Data Analytics Group (HDAG). HDAG is a non-profit student organization at Harvard College dedicated to helping organizations make smarter, data-driven decisions and achieve their strategic goals by translating data into meaningful and actionable insights.
    - Transition into a personalized summary of the recipient's company and role:
      > We understand that as the {title} at {company_name}, a leading organization in the {industry} space, you may face challenges in [specific area from "brief summary"]. HDAG is uniquely equipped to help you achieve your goals by leveraging our proven expertise and diverse project experience.

    ### 2. **Highlight a Relevant Case Study**
    - Build credibility by name-dropping HDAG's most notable clients:
      > HDAG has worked with a wide variety of clients across industries, including Coca-Cola, the World Health Organization, and Hewlett-Packard.
    - Introduce a single, highly relevant case study from the following examples:
      > {retrieved_cases}

    ### 3. **Call to Action**
    - Close with a polite and engaging call to action to encourage a response:
      > I would love to explore potential engagement opportunities between {company_name} and HDAG. Please let me know if you or a colleague would be available for a brief call in the next couple of business days to discuss how we can support your goals. I’m happy to accommodate any schedule that works best for you.
    - Ensure the tone remains conversational, polite, and professional, with a clear next step for the recipient.

    **Additional Notes:**
    - Keep the email concise (150-250 words) and focused on the recipient’s interests.
    - Use dynamic placeholders ({company_info} and {retrieved_cases}) to personalize the content.
    - Prioritize actionable and measurable results in the case study section.
    - Maintain a warm yet professional tone to maximize response rates.
    - Avoid overly technical language unless relevant to the recipient’s role or industry.
    """

    # Format the final prompt
    prompt = PromptTemplate(
        input_variables=["company_info", "retrieved_cases", "first_name", "title", "industry", "company_name", "sourcer_name"],
        template=prompt_template
    )
    final_prompt = prompt.format(
        company_info=company_text,
        retrieved_cases=retrieved_cases_text,
        first_name=first_name,
        title=title,
        industry=industry,
        company_name=company_name,
        sourcer_name=sourcer_name
    )

    # Generate email using LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    messages = [{"role": "system", "content": "You are a professional email writer."},
                {"role": "user", "content": final_prompt}]
    response = llm.invoke(messages)
    
    return response.content


# process csv of companies and generate emails for each company
def process_companies_and_generate_emails(companies_filepath, retriever, sourcer_name):
    companies_df = pd.read_csv(companies_filepath)

    emails = []
    for _, row in companies_df.iterrows():
        company_info = {
            "First Name": row.get("First Name", ""),
            "Last Name": row.get("Last Name", ""),
            "Title": row.get("Title", ""),
            "Company Name": row.get("Company Name for Emails", ""),
            "Industry": row.get("Industry", ""),
            "Technologies": row.get("Technologies", ""),
            "Keywords": row.get("Keywords", "")
        }

        email_content = generate_email(company_info, retriever, sourcer_name)
        emails.append({
            "Company Name": company_info["Company Name"],
            "Email Content": email_content
        })

    # Save to generated_emails.csv
    emails_df = pd.DataFrame(emails)
    emails_df.to_csv("generated_emails.csv", index=False)


def process_single_company(company_info, retriever):
    email_content = generate_email(company_info, retriever)
    print(f"Email for {company_info.get('Company Name', 'the company')}:\n")
    print(email_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate emails for companies.')
    parser.add_argument('--company', action='store_true', help='Generate email for a single company.')
    parser.add_argument('--company_name', type=str, help='Company Name')
    parser.add_argument('--industry', type=str, help='Industry')
    parser.add_argument('--technologies', type=str, help='Technologies')
    parser.add_argument('--keywords', type=str, help='Keywords')
    parser.add_argument('--companies_file', type=str, default='companies.csv', help='CSV file containing companies information')
    parser.add_argument('--sourcer_name', type=str, required=True, help="Sourcer's name for personalization")

    args = parser.parse_args()

    # temporary case data
    cases_df = load_and_process_data("case_data.csv")

    vectorstore = setup_pinecone_index(cases_df)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    if args.company:
        company_info = {
            "Company Name": args.company_name or "",
            "Industry": args.industry or "",
            "Technologies": args.technologies or "",
            "Keywords": args.keywords or ""
        }
        process_single_company(company_info, retriever, args.sourcer_name)
    else:
        process_companies_and_generate_emails(args.companies_file, retriever, args.sourcer_name)
