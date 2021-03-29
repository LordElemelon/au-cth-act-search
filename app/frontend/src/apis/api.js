import apiBase, { elasticClient } from ".";

class ApiService {
  fetchOriginalSection = async (names) => {
    const response = await apiBase.get("/read-sections", { names });
    if (response.status !== 200) throw new Error();
    const { result } = response.data.data;
    return result;
  };

  elasticsearch = async (query) => {
    const elasticQuery = {
      query: {
        match: {
          description: query,
        },
      },
    };
    const response = await elasticClient.get("/law/_search", elasticQuery);
    if (response.status !== 200) throw new Error();
    const { hits } = response.data.hits;
    return hits.map((hit) => hit._source.name);
  };

  basicSearch = async (query, technique) => {
    const response = await apiBase.get("/basic-search", { query, technique });
    if (response.status !== 200) throw new Error();
    const { result } = response.data.data;
    return result;
  };

  questionAnswering = async (question, technique) => {
    const response = await apiBase.get("/qa", { question, technique });
    if (response.status !== 200) throw new Error();
    const { answer } = response.data.data;
    return answer;
  };
}

export default new ApiService();
