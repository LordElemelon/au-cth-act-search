import axios from "axios";

const axiosApiClient = axios.create({
  baseURL: `http://7d802e23b32f.ngrok.io/`,
});

export const elasticClient = axios.create({
  baseURL: `http://localhost:9200/`,
});

export default axiosApiClient;
