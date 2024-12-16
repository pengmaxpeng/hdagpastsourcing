import os
import psutil
import pandas as pd
import glob
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

load_dotenv()

MEMORY_LIMIT_MB = 2000  # limit for memory usage per parsing operation

def parse_data_point_worker(file_path, data_point):
    parsing_instruction = f"""
        You are parsing a document related to a past sourcing project from the consulting group Harvard Data Analytics Group (HDAG). Extract only the following data point:

        **{data_point}**: Provide the requested information from the document.

        If the data point is in regards to a 'case', please only refer to the work done by the students in the organization as the 'case'. Do not describe the company in detail.
        For example, if the case is about a project done by students at Google, the 'case' would be the project done by HDAG, not the company Google.

        If you are unable to find a data point, please input 'UNKNOWN'.

        Also, use past tense in any long-length text outputs. For example, 'The students analyzed the data' instead of 'The students will analyze the data'.

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
    try:
        document = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
        if document:
            return document[0].text.strip()
    except Exception as e:
        print(f"Error occurred while parsing {file_path} for {data_point}: {e}")
    return ""

def infer_semester_from_path(file_path):
    """Infer the semester from the file path using a regex."""
    match = re.search(r'(F\d{2}|S\d{2})', file_path)
    if match:
        return match.group(0)
    return "Unknown"  # Default if no semester info is found

def parse_file(file_path, data_points):
    """Parse all data points for a single file."""
    result = {"file_path": file_path}
    result["Semester"] = infer_semester_from_path(file_path)  # Infer the semester
    for data_point in data_points:
        if data_point == "Semester":
            continue

        # Check memory usage before parsing
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        if memory_usage_mb > MEMORY_LIMIT_MB:
            print(f"Memory usage exceeded limit ({memory_usage_mb:.2f} MB) for {file_path}. Skipping.")
            return result  # Return partial results for this file
        
        # Parse the data point
        result[data_point] = parse_data_point_worker(file_path, data_point)
    return result

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_subdirs = ['cases', 'proposals']
    data_points = [
        "Case Name", "Company Name", "Industry", "Brief Case Description",
        "Detailed Case Description", "Outcome", "Programming Languages Used",
        "Tech Stack", "KPIs"
    ]

    all_files = []
    for subdir in data_subdirs:
        data_dir = os.path.join(current_dir, 'data', subdir)
        files = glob.glob(os.path.join(data_dir, "*"))
        all_files.extend(files)

    parsed_data_list = []
    max_workers = min(4, os.cpu_count() or 1)  # Use up to 4 threads or available CPUs
    print(f"Using up to {max_workers} workers for parallel parsing.")

    # Parallelize file parsing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(parse_file, file_path, data_points): file_path for file_path in all_files}

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                parsed_data = future.result()
                parsed_data_list.append(parsed_data)
                print(f"Finished parsing: {file_path}")
            except Exception as e:
                print(f"Error parsing file {file_path}: {e}")

    # Save results to CSV
    df = pd.DataFrame(parsed_data_list)
    df.to_csv('parsed_data3.csv', index=False)
    print("Data has been saved to parsed_data.csv")

if __name__ == '__main__':
    main()
