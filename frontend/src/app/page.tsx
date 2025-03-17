"use client"
import { useState } from "react";
import { generateDesign, analyzeImage } from "../../services/api";

export default function Home() {
  const [textPrompt, setTextPrompt] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  // This state allows for the description to be display for the image
  // const [description, setDescription] = useState("");
  const [analysis, setAnalysis] = useState("");

  const handleGenerate = async () => {
    const response = await generateDesign(textPrompt);
    setImageUrl(response.image_url);
    // setDescription(response.description);
  };

  const handleAnalyze = async () => {
    const response = await analyzeImage(imageUrl);
    setAnalysis(response.analysis);
  };

return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-center mb-4 text-gray-800">
          AI Architectural Design Assistant
        </h1>
        <div className="mb-4">
          <input
            type="text"
            className="w-full p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Describe your design..."
            onChange={(e) => setTextPrompt(e.target.value)}
          />
        </div>
        <div className="mb-4">
          <button
            onClick={handleGenerate}
            className="w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600 transition-colors"
          >
            Generate Design
          </button>
        </div>

        {imageUrl && (
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              Generated Image
            </h3>
            <img
              src={imageUrl}
              alt="AI-Generated Design"
              className="w-full h-auto rounded mb-2"
            />
            {/* <p className="text-gray-600 mb-2">
              <strong>Description:</strong> {description}
            </p> */}
            <button
              onClick={handleAnalyze}
              className="w-full bg-green-500 text-white py-2 px-4 rounded hover:bg-green-600 transition-colors"
            >
              Analyze Image
            </button>
          </div>
        )}

        {analysis && (
          <p className="text-gray-600 mt-4">
            <strong>Analysis:</strong> {analysis}
          </p>
        )}
      </div>
    </div>
  );
}
