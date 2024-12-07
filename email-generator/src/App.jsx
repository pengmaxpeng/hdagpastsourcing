import React, { useState } from "react";

function App() {
  const [formData, setFormData] = useState({
    company_name: "",
    industry: "",
    technologies: "",
    keywords: "",
  });
  const [emailContent, setEmailContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null); // For CSV file upload
  const [downloadLink, setDownloadLink] = useState(null); // For CSV download

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFileUpload = (e) => {
    setUploadedFile(e.target.files[0]);
  };

  const handleGenerateFromCsv = async () => {
    if (!uploadedFile) {
      alert("Please upload a .csv file first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await fetch("http://localhost:8000/generate-from-csv/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to generate emails from CSV");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadLink(url); // Generate link for downloading the CSV
    } catch (error) {
      console.error("Error generating emails from CSV:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const payload = {
      company_name: formData.company_name,
      industry: formData.industry,
      technologies: formData.technologies,
      keywords: formData.keywords,
    };

    console.log("Payload being sent to backend:", payload); // Debug log

    try {
      const response = await fetch("http://localhost:8000/generate-email/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Failed to generate email");
      }

      const data = await response.json();
      setEmailContent(data.email_content);
    } catch (error) {
      console.error("Error during email generation:", error);
      setEmailContent("Error generating email. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Email Generator</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Company Name:
          <input
            type="text"
            name="company_name"
            value={formData.company_name}
            onChange={handleChange}
          />
        </label>
        <label>
          Industry:
          <input
            type="text"
            name="industry"
            value={formData.industry}
            onChange={handleChange}
          />
        </label>
        <label>
          Technologies:
          <input
            type="text"
            name="technologies"
            value={formData.technologies}
            onChange={handleChange}
          />
        </label>
        <label>
          Keywords:
          <input
            type="text"
            name="keywords"
            value={formData.keywords}
            onChange={handleChange}
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Generating..." : "Generate Email"}
        </button>
      </form>

      <h2>Upload CSV for Batch Generation</h2>
      <input type="file" accept=".csv" onChange={handleFileUpload} />
      <button onClick={handleGenerateFromCsv} disabled={loading}>
        {loading ? "Processing..." : "Generate Emails from CSV"}
      </button>
      {downloadLink && (
        <div>
          <a href={downloadLink} download="generated_emails.csv">
            Download Generated Emails
          </a>
        </div>
      )}

      {emailContent && (
        <div>
          <h2>Generated Email</h2>
          <p>{emailContent}</p>
        </div>
      )}
    </div>
  );
}

export default App;
