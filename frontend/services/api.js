import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export const analyzeImage = async (file) => {
    const formData = new FormData();
    formData.append("image", file);
  
    const response = await axios.post(`${API_URL}/full-analyze`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });
  
    return response.data;
  };
  