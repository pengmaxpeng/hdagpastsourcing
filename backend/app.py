from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

from generate import load_and_process_data, setup_pinecone_index, generate_email

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retriever
cases_df = load_and_process_data("dummy_data.csv")
vectorstore = setup_pinecone_index(cases_df)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

class CompanyInfo(BaseModel):
    company_name: str
    industry: str
    technologies: str
    keywords: str

@app.post("/generate-email/")
async def generate_email_api(company_info: CompanyInfo):
    try:
        company_data = {
            "Company Name": company_info.company_name,
            "Industry": company_info.industry,
            "Technologies": company_info.technologies,
            "Keywords": company_info.keywords,
        }
        email_content = generate_email(company_data, retriever)
        return {"email_content": email_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-from-csv/")
async def generate_emails_from_csv(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)

        # Process the CSV to generate emails
        results = []
        for _, row in df.iterrows():
            company_data = {
                "Company Name": row.get("Company Name", ""),
                "Industry": row.get("Industry", ""),
                "Technologies": row.get("Technologies", ""),
                "Keywords": row.get("Keywords", ""),
            }
            email_content = generate_email(company_data, retriever)
            results.append({
                "Company Name": company_data["Company Name"],
                "Generated Email": email_content,
            })

        # Save the results to a new CSV file
        output_file = "generated_emails.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)

        return FileResponse(output_file, media_type="text/csv", filename=output_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
