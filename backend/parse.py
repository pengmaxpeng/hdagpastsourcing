from dotenv import load_dotenv
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import pandas as pd
import glob

load_dotenv()

print(os.getenv('LLAMA_API_KEY'))

def parse_data_point(file_path, data_point):
    parsing_instruction = f"""
        You are parsing a document related to a past sourcing project. Extract only the following data point:

        **{data_point}**: Provide the requested information from the document.

        Provide the extracted data as plain text, without any additional text or formatting.
    """
    parser = LlamaParse(
        result_type="markdown",  # Use 'markdown' as it's supported
        api_key=os.getenv('LLAMA_API_KEY'),
        parsing_instruction=parsing_instruction,
        chunk_size=10000,
    )

    # Use LlamaParse as a file_extractor in SimpleDirectoryReader
    file_extractor = {".pdf": parser}
    document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

    if not document:
        print(f"Failed to parse {file_path}, skipping.")
        return ""
    
    # Access the parsed result from document[0].text
    result = document[0].text.strip()
    return result

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_subdirs = ['cases', 'presentations', 'proposals']
    data_points = [
        "Case Name", "Semester", "Company Name", "Brief Case Description",
        "Detailed Case Description", "Outcome", "Programming Languages Used",
        "Tech Stack", "KPIs"
    ]

    # Initialize the data structure to hold data for each file
    file_data = {}  # Key: file_path, Value: dict of data points

    all_files = []

    for subdir in data_subdirs:
        data_dir = os.path.join(current_dir, 'data', subdir)
        files = glob.glob(os.path.join(data_dir, "*"))
        all_files.extend(files)

    # Initialize file_data with file paths
    for file_num, file_path in enumerate(all_files, start=1):
        file_data[file_path] = {"file_num": file_num}

    # Now, for each data point, parse all files
    for data_point in data_points:
        print(f"Parsing data point: {data_point}")
        for file_path in file_data.keys():
            parsed_value = parse_data_point(file_path, data_point)
            file_data[file_path][data_point] = parsed_value

    # Convert the data into a list of dictionaries for DataFrame
    parsed_data_list = list(file_data.values())

    # Save results to CSV
    df = pd.DataFrame(parsed_data_list)
    df.to_csv('parsed_data.csv', index=False)
    print("Data has been saved to parsed_data.csv")

if __name__ == '__main__':
    main()
