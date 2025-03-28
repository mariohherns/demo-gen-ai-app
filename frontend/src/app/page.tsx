"use client";

import { useState } from "react";
import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export default function Home() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setResult(null); // Clear result
    }
  };

  const handleAnalyze = async () => {
    if (!image) return;

    const formData = new FormData();
    formData.append("image", image);

    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/full-analyze`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      console.log(response.data);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Failed to analyze image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <div className="w-full max-w-md bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          Chest X-ray Analyzer
        </h1>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="mb-4 w-full text-gray-700"
        />

        {preview && (
          <img
            src={preview}
            alt="Preview"
            className="rounded w-full h-auto mb-4 border"
          />
        )}

        <button
          onClick={handleAnalyze}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors mb-4"
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze X-ray"}
        </button>

        {result && (
          <div className="text-gray-700 space-y-2">
            <h2 className="text-lg font-semibold">Predictions:</h2>
            <ul className="list-disc pl-6">
              {Object.entries(result.predictions).map(([label, confidence]: any) => (
                <li key={label}>
                  <strong className="capitalize">{label}</strong>: {(confidence * 100).toFixed(2)}%
                </li>
              ))}
            </ul>
            <p className="mt-4">
              <strong>Explanation:</strong> {result.explanation}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}



  
