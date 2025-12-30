import torch
import torch.nn.functional as F
import pickle

from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Tuple

from src.data_utils import load_json_file
from src.search.simple_encoder import SimpleEncoder
from src.search.logger_config import logger


class E5BaseSearcher:
    def __init__(
            self,
            dataset_name: str,
            index_version: str='v0',
            device: str='cuda',
            verbose: bool=False
    ):
        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.verbose = verbose

        n_gpus: int = torch.cuda.device_count()
        self.gpu_ids: List[int] = list(range(n_gpus))

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path='intfloat/e5-base-v2',
            max_length=512,
        )
        self.encoder.to(self.gpu_ids[-1])

        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.index_path = f"index/{dataset_name}/e5_base_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            self.index_path = f"index/{dataset_name}/e5_base_{dataset_name}.pkl"

        with open(self.index_path, "rb") as f:
            _, all_embeddings = pickle.load(f)
        all_embeddings = torch.tensor(all_embeddings)

        logger.info(f'Load {all_embeddings.shape[0]} embeddings from {self.index_path}')

        split_embeddings = torch.chunk(all_embeddings, len(self.gpu_ids))
        self.embeddings: List[torch.Tensor] = [
            split_embeddings[i].to(f'cuda:{self.gpu_ids[i]}', dtype=torch.float16) for i in range(len(self.gpu_ids))
        ]

    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        query_embed: torch.Tensor = self.encoder.encode_queries(queries).to(dtype=self.embeddings[0].dtype)

        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(queries)):
            results: List[Dict] = []
            for score, idx in zip(batch_sorted_score[query_idx], batch_sorted_indices[query_idx]):
                results.append({
                    'doc_id': int(idx.item()),
                    'score': score.item(),
                })

                if self.verbose:
                    results[-1].update(self.corpus[int(idx.item())])
            results_list.append(results)
        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """
        [
                [ # 첫 번째 쿼리 결과
                    {'doc_id': 409, 'score': 0.86474609375},
                    {'doc_id': 896, 'score': 0.86376953125},
                    {'doc_id': 894, 'score': 0.86328125},
                    {'doc_id': 1794, 'score': 0.85986328125},
                    {'doc_id': 895, 'score': 0.859375}
                ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]
        """
        docs: List[List[str]] = []
        for idx, results in enumerate(results_list):
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def _compute_topk(self, query_embed: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_score_list: List[torch.Tensor] = []
        batch_sorted_indices_list: List[torch.Tensor] = []

        idx_offset = 0
        for i in range(len(self.embeddings)):
            query_embed = query_embed.to(self.embeddings[i].device)
            score = torch.mm(query_embed, self.embeddings[i].t())
            sorted_score, sorted_indices = torch.topk(score, k=k, dim=-1, largest=True)

            sorted_indices += idx_offset
            batch_score_list.append(sorted_score.cpu())
            batch_sorted_indices_list.append(sorted_indices.cpu())
            idx_offset += self.embeddings[i].shape[0]

        batch_score = torch.cat(batch_score_list, dim=1)
        batch_sorted_indices = torch.cat(batch_sorted_indices_list, dim=1)
        
        # only keep the top k results based on batch_score
        batch_score, top_indices = torch.topk(batch_score, k=k, dim=-1, largest=True)
        batch_sorted_indices = torch.gather(batch_sorted_indices, dim=1, index=top_indices)

        return batch_score, batch_sorted_indices


class E5LargeSearcher:
    def __init__(
            self,
            dataset_name: str,
            index_version: str='v0',
            device: str='cuda',
            verbose: bool=False
    ):
        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        self.verbose = verbose

        n_gpus: int = torch.cuda.device_count()
        self.gpu_ids: List[int] = list(range(n_gpus))

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path='intfloat/e5-large-v2',
            max_length=512,
        )
        self.encoder.to(self.gpu_ids[-1])

        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.index_path = f"index/{dataset_name}/e5_large_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            self.index_path = f"index/{dataset_name}/dense/e5_large_{dataset_name}.pkl"
        
        with open(self.index_path, "rb") as f:
            _, all_embeddings = pickle.load(f)
        all_embeddings = torch.tensor(all_embeddings)

        logger.info(f'Load {all_embeddings.shape[0]} embeddings from {self.index_path}')

        split_embeddings = torch.chunk(all_embeddings, len(self.gpu_ids))
        self.embeddings: List[torch.Tensor] = [
            split_embeddings[i].to(f'cuda:{self.gpu_ids[i]}', dtype=torch.float16) for i in range(len(self.gpu_ids))
        ]

    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        query_embed: torch.Tensor = self.encoder.encode_queries(queries).to(dtype=self.embeddings[0].dtype)

        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(queries)):
            results: List[Dict] = []
            for score, idx in zip(batch_sorted_score[query_idx], batch_sorted_indices[query_idx]):
                results.append({
                    'doc_id': int(idx.item()),
                    'score': score.item(),
                })

                if self.verbose:
                    results[-1].update(self.corpus[int(idx.item())])
            results_list.append(results)
        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """
        [
                [ # 첫 번째 쿼리 결과
                    {'doc_id': 409, 'score': 0.86474609375},
                    {'doc_id': 896, 'score': 0.86376953125},
                    {'doc_id': 894, 'score': 0.86328125},
                    {'doc_id': 1794, 'score': 0.85986328125},
                    {'doc_id': 895, 'score': 0.859375}
                ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]
        """
        docs: List[List[str]] = []
        for idx, results in enumerate(results_list):
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def _compute_topk(self, query_embed: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_score_list: List[torch.Tensor] = []
        batch_sorted_indices_list: List[torch.Tensor] = []

        idx_offset = 0
        for i in range(len(self.embeddings)):
            query_embed = query_embed.to(self.embeddings[i].device)
            score = torch.mm(query_embed, self.embeddings[i].t())
            sorted_score, sorted_indices = torch.topk(score, k=k, dim=-1, largest=True)

            sorted_indices += idx_offset
            batch_score_list.append(sorted_score.cpu())
            batch_sorted_indices_list.append(sorted_indices.cpu())
            idx_offset += self.embeddings[i].shape[0]

        batch_score = torch.cat(batch_score_list, dim=1)
        batch_sorted_indices = torch.cat(batch_sorted_indices_list, dim=1)
        
        # only keep the top k results based on batch_score
        batch_score, top_indices = torch.topk(batch_score, k=k, dim=-1, largest=True)
        batch_sorted_indices = torch.gather(batch_sorted_indices, dim=1, index=top_indices)

        return batch_score, batch_sorted_indices


class Contriever:
    def __init__(
            self,
            index_version: str,
            dataset: str,
            model_name_or_path: str='facebook/contriever',
            device: str=None
    ):
        # load corpus
        self.corpus = load_json_file(f"data/{dataset}/norms.json")

        # load embeddings
        embedding_path = f"data/index/{dataset}/contriever_{dataset}_{index_version}.pkl"
        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        self.ids = all_ids

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float16).to(self.device)
        print(f"Loading embeddings with dtype: {self.embeddings.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.model.to(self.device).eval()

    # Mean pooling
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def encode_queries(self, queries: List[str], max_length: int = 512) -> torch.Tensor:
        # tokenize queries
        queries_tokenized = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            # query embeddings
            query_embeddings = self.model(**queries_tokenized)
            query_embeddings = self.mean_pooling(query_embeddings[0], queries_tokenized['attention_mask'])

            # Normalize embeddings
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1) # L2 norm ,, 일단 해보자

        return query_embeddings

    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings = self.encode_queries(queries)  # shape: (num_queries, dim)
        # query_embs = query_embeddings.to(dtype=self.embeddings.dtype, device=self.embeddings.device)
            
        # BUG FIX: `self.embeddings`가 단일 텐서이므로 행렬곱이 정상적으로 동작
        all_scores = query_embeddings @ self.embeddings.T

        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        # 결과를 CPU로 이동
        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        results_list = []
        for i in range(len(queries)):
            query_results = []
            for j in range(k):
                doc_index = top_k_indices[i, j]
                score = top_k_scores[i, j]
                # doc_id를 self.ids에서 직접 조회
                doc_id = self.ids[doc_index] 
                query_results.append({'doc_id': int(doc_id), 'score': float(score)})
            results_list.append(query_results)

        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
            self,
            queries: List[str],
            query_ids: List[str],
            k: int
    ) -> List[Tuple[List[str], str]]:
        """
        Return Examples
        [
            ([pred1, pred2, pred3], gold),
            ([pred1, pred2, pred3], gold),
            ...
        ]
        Preserves the order of provided query_ids.
        """
        results_list: List[List[Dict]] = self.batch_search(queries, top_k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        return docs


class BGELargeENSearcher:
    def __init__(
            self,
            index_version: str,
            dataset: str,
            model_name_or_path: str='BAAI/bge-large-en-v1.5',
            device: str = None
    ):
        # load corpus
        if (dataset == "normad") or (dataset == "scruples"):
            self.corpus = load_json_file(f"data/{dataset}/norms.json")
        elif dataset == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # load embeddings
        embedding_path = f"data/index/{dataset}/bge_large_en_{dataset}_{index_version}.pkl"
        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        self.ids = all_ids  # <-- BUG FIX: self.ids 할당

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embeddings in FP16 directly
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float16).to(self.device)
        print(f"Loading embeddings with dtype: {self.embeddings.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
        self.model.to(self.device).eval()
        
        # Set batch size for memory efficiency
        self.batch_size = 32
    

    def encode_queries(self, queries: List[str], max_length: int = 512) -> torch.Tensor:
        queries_with_instruction = [f"Represent this sentence for searching relevant passages: {query}" for query in queries]
        
        all_embeddings = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(queries_with_instruction), self.batch_size):
            batch_queries = queries_with_instruction[i:i + self.batch_size]
            
            # tokenize queries
            query_tokenized = self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                query_embeddings = self.model(**query_tokenized)

                # CLS pooling
                query_embeddings = query_embeddings.last_hidden_state[:, 0]

                # Normalize
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                
                all_embeddings.append(query_embeddings)
        
        return torch.cat(all_embeddings, dim=0)

    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings = self.encode_queries(queries)  # shape: (num_queries, dim)
        # query_embs = query_embs.to(dtype=self.embeddings.dtype, device=self.embeddings.device)

        # BUG FIX: `self.embeddings`가 단일 텐서이므로 행렬곱이 정상적으로 동작
        all_scores = query_embeddings @ self.embeddings.T

        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        # 결과를 CPU로 이동하고 메모리 정리
        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        
        # GPU 메모리 정리
        del query_embeddings, all_scores
        torch.cuda.empty_cache()

        results_list = []
        for i in range(len(queries)):
            query_results = []
            for j in range(k):
                doc_index = top_k_indices[i, j]
                score = top_k_scores[i, j]
                # doc_id를 self.ids에서 직접 조회
                doc_id = self.ids[doc_index] 
                query_results.append({'doc_id': int(doc_id), 'score': float(score)})
            results_list.append(query_results)

        return results_list
    
    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
            self,
            queries: List[str],
            query_ids: List[str],
            k: int
    ) -> List[Tuple[List[str], str]]:
        """
        Return Examples
        [
            ([pred1, pred2, pred3], gold),
            ([pred1, pred2, pred3], gold),
            ...
        ]
        Preserves the order of provided query_ids.
        """
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        return docs


class BGEM3Searcher:
    def __init__(
        self,
        dataset_name: str,
        index_version: str='v0',
        device: str=None
    ):
        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            embedding_path = f"index/{dataset_name}/bge_m3_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            embedding_path = f"index/{dataset_name}/bge_m3_{dataset_name}.pkl"
        
        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        
        self.ids = all_ids
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # convert embeddings to PyTorch tensor and move to specified device
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float16, device=self.device)
        print(f"Loading embeddings with dtype: {self.embeddings.dtype}")

        # initialize BGEM3FlagModel
        self.model = BGEM3FlagModel(
            'BAAI/bge-m3',
            use_fp16=True, # use fp16 for faster inference
        )
        print(f"Load {self.embeddings.shape[0]} embeddings from {embedding_path}")

    def _encode_queries(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        query_embeddings = self.model.encode(
            texts,
            batch_size=128,
            max_length=max_length,
            return_dense=True, return_sparse=False, return_colbert_vecs=False
        )['dense_vecs'] # float16, L2-normalized
        
        query_embeddings = torch.from_numpy(query_embeddings).to(self.device)
        return query_embeddings

    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings = self._encode_queries(queries)

        # 텍스트가 그렇게 많지 않기 때문에, 메모리 여유가 있어서 그냥 한 번에 하는 걸로,,
        all_scores = query_embeddings @ self.embeddings.T
        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        results_list = []
        for i in range(len(queries)):
            query_results = []
            for j in range(k):
                doc_index = top_k_indices[i, j]
                score = top_k_scores[i, j]
                doc_id = self.ids[doc_index]
                query_results.append({'doc_id': int(doc_id), 'score': float(score)})
            results_list.append(query_results)
        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
        self,
        queries: List[str],
        query_ids: List[str],
        k: int
    ) -> List[Tuple[List[str], str]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        
        return docs


class LLMEmbedderSearcher:
    def __init__(
            self,
            dataset_name: str,
            index_version: str='v0',
            shot: str='3shot',
            device: str=None
    ):
        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/{shot}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            embedding_path = f"index/{dataset_name}/dense/{shot}/llm_embedder_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            embedding_path = f"index/{dataset_name}/dense/llm_embedder_{dataset_name}.pkl"
        
        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        self.ids = all_ids

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.embeddings = torch.tensor(all_embeddings, dtype=torch.float16).to(self.device)
        print(f"Loading embeddings with dtype: {self.embeddings.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/llm-embedder')
        self.model = AutoModel.from_pretrained('BAAI/llm-embedder', torch_dtype=torch.float16)
        self.model.to(self.device).eval()

        self.instruction = "Represent this query for retrieval: "

    def _encode_queries(self, queries: List[str], max_length: int = 512) -> torch.Tensor:
        queries_with_instruction = [f"{self.instruction}{query}" for query in queries]
        
        # tokenize queries
        query_tokenized = self.tokenizer(
            queries_with_instruction,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            query_embeddings = self.model(**query_tokenized)

            # CLS pooling
            query_embeddings = query_embeddings.last_hidden_state[:, 0]

            # Normalize
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return query_embeddings

    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings = self._encode_queries(queries)  # shape: (num_queries, dim)
        # query_embeddings = query_embeddings.to(dtype=self.embeddings.dtype, device=self.embeddings.device)

        # BUG FIX: `self.embeddings`가 단일 텐서이므로 행렬곱이 정상적으로 동작
        all_scores = query_embeddings @ self.embeddings.T

        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        # 결과를 CPU로 이동
        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        results_list = []
        for i in range(len(queries)):
            query_results = []
            for j in range(k):
                doc_index = top_k_indices[i, j]
                score = top_k_scores[i, j]
                # doc_id를 self.ids에서 직접 조회
                doc_id = self.ids[doc_index] 
                query_results.append({'doc_id': int(doc_id), 'score': float(score)})
            results_list.append(query_results)

        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
            self,
            queries: List[str],
            query_ids: List[str],
            k: int
    ) -> List[Tuple[List[str], str]]:
        """
        Return Examples
        [
            ([pred1, pred2, pred3], gold),
            ([pred1, pred2, pred3], gold),
            ...
        ]
        Preserves the order of provided query_ids.
        """
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        return docs


class MiniLML6Searcher:
    def __init__(
            self,
            dataset_name: str,
            index_version: str='v0',
            shot: str='3shot',
            device: str=None
    ):
        # set device first
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/{shot}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            embedding_path = f"index/{dataset_name}/dense/{shot}/miniLM_L6_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            embedding_path = f"index/{dataset_name}/dense/miniLM_L6_{dataset_name}.pkl"

        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        self.ids = all_ids
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading embeddings with dtype: {dtype}")
        self.embeddings = torch.tensor(all_embeddings, dtype=dtype).to(self.device)
        print(f"Load {self.embeddings.shape[0]} embeddings from {embedding_path}")

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.to(self.device).eval().half() # use fp16 for faster inference

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_queries(self, texts: List[str], max_length: int=512) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings: torch.Tensor = self._encode_queries(queries)
        query_embeddings = query_embeddings.to(dtype=self.embeddings.dtype, device=self.embeddings.device)

        all_scores = query_embeddings @ self.embeddings.T
        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        results_list: List[List[Dict]] = []
        for i in range(len(queries)):
            results: List[Dict] = []
            for j in range(k):
                doc_index = int(top_k_indices[i, j])
                score = float(top_k_scores[i, j])
                doc_id = self.ids[doc_index]
                results.append({'doc_id': int(doc_id), 'score': score})
            results_list.append(results)
        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
            self,
            queries: List[str],
            query_ids: List[str],
            k: int
    ) -> List[Tuple[List[str], str]]:
        """
        Return Examples
        [
            ([pred1, pred2, pred3], gold),
            ([pred1, pred2, pred3], gold),
            ...
        ]
        Preserves the order of provided query_ids.
        """
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        return docs


class MiniLML12Searcher:
    def __init__(
            self,
            dataset_name: str,
            index_version: str='v0',
            shot: str='3shot',
            device: str=None
    ):
        # set device first
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # load corpus
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            self.corpus = load_json_file(f"data/{dataset_name}/{shot}/norms.json")
        elif dataset_name == "normad_region":
            self.corpus = load_json_file(f"data/normad/{dataset_name}_norms.json")
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # load embeddings
        if (dataset_name == "normad") or (dataset_name == "scruples"):
            embedding_path = f"index/{dataset_name}/dense/{shot}/miniLM_L12_{dataset_name}_{index_version}.pkl"
        elif dataset_name == "normad_region":
            embedding_path = f"index/{dataset_name}/dense/miniLM_L12_{dataset_name}.pkl"

        with open(embedding_path, "rb") as f:
            all_ids, all_embeddings = pickle.load(f)
        self.ids = all_ids
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading embeddings with dtype: {dtype}")
        self.embeddings = torch.tensor(all_embeddings, dtype=dtype).to(self.device)
        print(f"Load {self.embeddings.shape[0]} embeddings from {embedding_path}")

        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        self.model.to(self.device).eval().half() # use fp16 for faster inference

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _encode_queries(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded)
            embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def batch_search(self, queries: List[str], k: int) -> List[List[Dict]]:
        query_embeddings: torch.Tensor = self._encode_queries(queries)
        query_embeddings = query_embeddings.to(dtype=self.embeddings.dtype, device=self.embeddings.device)

        all_scores = query_embeddings @ self.embeddings.T
        top_k_scores, top_k_indices = torch.topk(all_scores, k=k, dim=1)

        top_k_scores = top_k_scores.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()

        results_list: List[List[Dict]] = []
        for i in range(len(queries)):
            results: List[Dict] = []
            for j in range(k):
                doc_index = int(top_k_indices[i, j])
                score = float(top_k_scores[i, j])
                doc_id = self.ids[doc_index]
                results.append({'doc_id': int(doc_id), 'score': score})
            results_list.append(results)
        return results_list

    def batch_search_with_docs(self, queries: List[str], k: int) -> List[List[Dict]]:
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)
        """[
            [
                {'doc_id': 409, 'score': 0.86474609375},
                {'doc_id': 896, 'score': 0.86376953125},
                {'doc_id': 894, 'score': 0.86328125},
                {'doc_id': 1794, 'score': 0.85986328125},
                {'doc_id': 895, 'score': 0.859375}
            ],
            ... --> 이런 게 쿼리 수만큼 있음
        ]"""
        docs: List[List[str]] = []
        for results in results_list:
            docs.append([self.corpus[int(result['doc_id'])]['rot'] for result in results])
        return docs

    def batch_search_docs_with_gold(
            self,
            queries: List[str],
            query_ids: List[str],
            k: int
    ) -> List[Tuple[List[str], str]]:
        """
        Return Examples
        [
            ([pred1, pred2, pred3], gold),
            ([pred1, pred2, pred3], gold),
            ...
        ]
        Preserves the order of provided query_ids.
        """
        results_list: List[List[Dict]] = self.batch_search(queries, k=k)

        docs: List[Tuple[List[str], str]] = []
        for i, qid in enumerate(query_ids):
            preds: List[str] = [self.corpus[int(r['doc_id'])]['rot'] for r in results_list[i]]
            gold: str = self.corpus[int(qid)]['rot']
            docs.append((preds, gold))
        return docs