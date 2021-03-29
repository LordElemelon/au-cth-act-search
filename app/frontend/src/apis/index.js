import axios from "axios";

const axiosApiClient = axios.create({
  baseURL: `http://localhost:5000/`,
});

export const elasticClient = axios.create({
  baseURL: `http://localhost:9200/`,
});

export default axiosApiClient;
