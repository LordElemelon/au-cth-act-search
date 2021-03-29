import apiBase, { elasticClient } from ".";

class ApiService {
  fetchOriginalSection = async (names) => {
    console.log(names);
    const response = await apiBase.post("/read-sections", { names });
    if (response.status !== 200) throw new Error();
    const { result } = response.data;
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
    const response = await elasticClient.post("/law/_search", elasticQuery);
    if (response.status !== 200) throw new Error();
    const { hits } = response.data.hits;
    return hits.map((hit) => hit._source.name);
  };

  basicSearch = async (query, technique) => {
    const response = await apiBase.post("/basic-search", { query, technique });
    if (response.status !== 200) throw new Error();
    const { result } = response.data;
    return result;
  };

  questionAnswering = async (question, technique) => {
    const response = await apiBase.post("/qa", { question, technique });
    if (response.status !== 200) throw new Error();
    const { answer } = response.data;
    return answer;
  };
}

export default new ApiService();
