# flake8: noqa: E402  — sys.path 操作必须在本地导入之前
"""RAGAS 评估脚本 — 通过完整 8-Agent 图执行评估。

覆盖全链路：Orchestrator → Intent → Retriever → DocFilter
           → ContextCompress → Reason → Writer → AntiHallucination
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# 必须在本地导入前设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from app.config import settings
from app.core.retriever import HybridRetriever
from app.graph import build_graph
from app.utils.logger import get_logger

# HuggingFace 镜像 & 离线模式（必须在 app 模块导入后设置）
if settings.HF_ENDPOINT:
    os.environ['HF_ENDPOINT'] = settings.HF_ENDPOINT
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

logger = get_logger(__name__)

# 尝试导入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class RagasEvaluator:
    """通过完整 8-Agent 图的 RAGAS 评估器。"""

    CACHE_FILE = os.path.join(os.path.dirname(__file__), "rag_generation_cache.json")

    def __init__(self):
        logger.info("Initializing RAGAS Evaluator (8-Agent mode)...")

        # 1. 初始化底层组件
        self.retriever = HybridRetriever()

        # 2. 构建完整 8-Agent 图
        logger.info("Building 8-Agent graph for evaluation...")
        self.graph = build_graph(self.retriever)
        logger.info("8-Agent graph ready.")

        # 3. RAGAS 评估 LLM
        self.eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model_name=settings.LLM_MODEL_NAME,
                openai_api_key=settings.DEEPSEEK_API_KEY,
                openai_api_base=settings.DEEPSEEK_BASE_URL,
                temperature=0.0,
            )
        )

        # 4. RAGAS 评估 Embedding
        model_path = settings.LOCAL_EMBEDDING_PATH
        logger.info(f"Loading evaluation embedding from: {model_path}")
        embed_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'},
        )
        self.eval_embeddings = LangchainEmbeddingsWrapper(embed_model)

    async def run_pipeline_async(self, question: str) -> Dict[str, Any]:
        """通过完整 8-Agent 图运行一次问答。

        流程：orchestrator → intent → retriever → doc_filter
              → context_compress → reason → writer → anti_hallucination
        """
        try:
            # 执行完整 LangGraph 工作流
            result = await self.graph.ainvoke(
                {
                    "question": question,
                    "session_id": "",  # 评估模式不使用对话记忆
                },
                config={"recursion_limit": 50},
            )

            answer = result.get("final_answer", "")
            documents = result.get("raw_docs", [])
            node_log = result.get("node_log", [])
            hallucination_risk = result.get("hallucination_risk", "")
            trace_id = result.get("trace_id", "")

            # 提取文本作为 RAGAS contexts
            contexts = [doc.get("text", "") for doc in documents if doc.get("text")]

            logger.info(
                "Graph complete: trace=%s answer=%d chars, "
                "contexts=%d, verdict=%s, path=%s",
                trace_id[:8] if trace_id else "none",
                len(answer),
                len(contexts),
                hallucination_risk,
                " → ".join(node_log[-4:]),
            )

            return {
                "question": question,
                "answer": answer or "未生成答案",
                "contexts": contexts,
                "success": bool(answer),
                "hallucination_risk": hallucination_risk,
                "node_log": node_log,
            }

        except Exception as e:
            logger.error(
                "Graph pipeline error for '%s...': %s",
                question[:50],
                e,
                exc_info=True,
            )
            return {
                "question": question,
                "answer": f"Graph Error: {str(e)}",
                "contexts": [],
                "success": False,
                "hallucination_risk": "error",
                "node_log": [],
            }

    def load_or_prepare_dataset(
        self,
        test_cases: List[Dict[str, str]],
        max_concurrency: int = 3,
    ) -> Dataset:
        """准备 RAGAS 数据集，支持缓存。"""
        cache_path = Path(self.CACHE_FILE)

        # 1. 尝试缓存
        if cache_path.exists():
            logger.info(f"Loading cached results from {self.CACHE_FILE}...")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)

                if len(cached_data) == len(test_cases):
                    questions = [item['question'] for item in cached_data]
                    answers = [item['answer'] for item in cached_data]
                    contexts = [item['contexts'] for item in cached_data]
                    ground_truths = [item['ground_truth'] for item in cached_data]

                    logger.info("Cache loaded successfully.")
                    return Dataset.from_dict({
                        'question': questions,
                        'answer': answers,
                        'contexts': contexts,
                        'ground_truth': ground_truths,
                    })
                else:
                    logger.warning("Cache size mismatch. Regenerating...")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}. Regenerating...")

        # 2. 无缓存 → 通过图生成
        logger.info(
            f"Generating answers for {len(test_cases)} cases "
            f"(concurrency={max_concurrency}) via 8-agent graph..."
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_case(case: Dict[str, str]):
            async with semaphore:
                result = await self.run_pipeline_async(case["question"])
                if result["success"]:
                    return {
                        "question": case["question"],
                        "answer": result["answer"],
                        "ground_truth": case["ground_truth"],
                        "contexts": result["contexts"],
                        "_hallucination_risk": result.get("hallucination_risk", ""),
                        "_node_log": result.get("node_log", []),
                    }
                else:
                    logger.warning(f"Skipping failed case: {case['question'][:50]}...")
                    return None

        async def run_all():
            tasks = [process_case(c) for c in test_cases]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            valid_results = loop.run_until_complete(run_all())
        except RuntimeError:
            import nest_asyncio
            nest_asyncio.apply()
            valid_results = asyncio.run(run_all())

        if not valid_results:
            logger.error("No valid results generated.")
            return Dataset.from_dict({
                'question': [], 'answer': [], 'contexts': [], 'ground_truth': [],
            })

        questions = [r["question"] for r in valid_results]
        answers = [r["answer"] for r in valid_results]
        ground_truths = [r["ground_truth"] for r in valid_results]
        contexts = [r["contexts"] for r in valid_results]

        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths,
        })

        # 3. 保存缓存
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(valid_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results cached to {self.CACHE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return dataset

    def evaluate_system(self, test_cases: List[Dict[str, str]]):
        """执行完整评估。"""
        dataset = self.load_or_prepare_dataset(test_cases)

        if len(dataset) == 0:
            logger.error("Dataset is empty. Cannot evaluate.")
            return None

        logger.info("Starting RAGAS evaluation (this may take a while)...")

        metrics = [
            faithfulness,       # 忠实度：回答是否完全依托检索上下文，无编造幻觉
            answer_relevancy,   # 回答相关性：回答和用户提问是否匹配
            context_precision,  # 上下文精确率：检索到的片段里有效有用信息占比
            context_recall,     # 上下文召回率：问题所需关键信息是否全被检索出来
            answer_correctness, # 回答正确率：回答事实、逻辑是否准确无误
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embeddings,
            raise_exceptions=False,
        )

        print("\n" + "=" * 50)
        print("  RAGAS Evaluation Results (8-Agent Graph)")
        print("=" * 50)
        df_results = result.to_pandas()
        print(df_results.describe())
        print("\n--- Per-question scores ---")
        print(df_results)

        output_file = os.path.join(
            os.path.dirname(__file__), "ragas_evaluation_results.csv"
        )
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        return result


# ====================================================================
# 主入口
# ====================================================================

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(__file__), "test_dataset.csv")

    if not os.path.exists(csv_path):
        logger.error(f"Test dataset not found: {csv_path}")
        logger.info(
            "Create test_dataset.csv with 'question' and 'ground_truth' columns."
        )
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        if 'question' not in df.columns or 'ground_truth' not in df.columns:
            raise ValueError("CSV must contain 'question' and 'ground_truth' columns")
        TEST_CASES = df.to_dict(orient='records')
        logger.info(f"Loaded {len(TEST_CASES)} test cases from CSV")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        sys.exit(1)

    evaluator = RagasEvaluator()
    logger.info("Starting RAGAS evaluation via 8-agent graph...")
    evaluator.evaluate_system(TEST_CASES)
