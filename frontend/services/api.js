import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export const generateDesign = async (text) => {
  const response = await axios.post(`${API_URL}/generate`, { text });
  return response.data;
};

export const analyzeImage = async (imageUrl) => {
  const response = await axios.post(`${API_URL}/analyze-image`, {
    image_url: imageUrl,
  });
  return response.data;
};
