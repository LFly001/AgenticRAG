"""文档解析器 — 基于 Unstructured 库解析 PDF / DOCX 并提取结构化元素。"""
import os
import sys
import re
import nltk
from typing import List, Dict, Any

from app.config import settings
from app.utils.logger import get_logger

from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.documents.elements import Table


def _get_nltk_data_path():
    """动态获取 NLTK 数据路径，兼容不同操作系统。"""
    env_path = os.environ.get('NLTK_DATA')
    if env_path and os.path.exists(env_path):
        return env_path

    local_path = os.path.join(os.getcwd(), 'nltk_data')
    if os.path.exists(local_path):
        return local_path

    user_home = os.path.expanduser("~")
    default_path = os.path.join(user_home, 'nltk_data')
    return default_path


NLTK_DATA_PATH = _get_nltk_data_path()

if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)


def _init_nltk_resources():
    required_resources = [
        'tokenizers/punkt_tab',
        'tokenizers/punkt',
        'taggers/averaged_perceptron_tagger_eng',
        'corpora/stopwords',
    ]

    missing_resources = []
    for res in required_resources:
        try:
            nltk.data.find(res)
        except LookupError:
            missing_resources.append(res)

    if missing_resources:
        error_msg = (
            f"[NLTK Init Error] Missing resources: {', '.join(missing_resources)}. "
            f"Please download them to {NLTK_DATA_PATH} manually."
        )
        print(error_msg, file=sys.stderr)


_init_nltk_resources()

os.environ['NLTK_DOWNLOAD_DIR'] = NLTK_DATA_PATH

logger = get_logger(__name__)


class DocumentParser:
    @staticmethod
    def parse_file(file_path: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(file_path)[1].lower()
        elements = []

        try:
            strategy = getattr(settings, 'PARSING_STRATEGY', "auto")
            logger.info(f"Parsing {file_path} with strategy: {strategy}")

            if ext == ".pdf":
                elements = partition_pdf(
                    filename=file_path,
                    strategy=strategy,
                    infer_table_structure=True,  # 开启表格结构识别
                    chunking_strategy=None,      # 保持原始元素，由后续 SmartChunker 处理
                    languages=["eng", "chi_sim"]  # 根据需求添加语言支持，提升 OCR 准确率
                )
            elif ext == ".docx":
                elements = partition_docx(
                    filename=file_path,
                    infer_table_structure=True,
                )
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            return DocumentParser._convert_elements_to_dicts(elements, file_path)

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}", exc_info=True)
            return []

    @staticmethod
    def _convert_elements_to_dicts(elements: List[Any], source_file: str) -> List[Dict[str, Any]]:
        processed_data = []

        # 只保留文件名，避免完整路径在 URL 传参时被规范化导致匹配失败
        clean_source = os.path.basename(source_file)

        for elem in elements:
            if not hasattr(elem, 'text') or not elem.text:
                continue

            text_content = elem.text.strip()
            if len(text_content) < 5:
                continue

            metadata = getattr(elem, 'metadata', None)
            page_number = getattr(metadata, 'page_number', None) if metadata else None
            category = getattr(elem, 'category', 'Unknown')

            final_text = text_content
            is_table = False

            if category == "Table" or isinstance(elem, Table):
                is_table = True
                if metadata and hasattr(metadata, 'text_as_html') and metadata.text_as_html:
                    final_text = DocumentParser._html_to_markdown(metadata.text_as_html)
                else:
                    final_text = f"[Table Content]\n{text_content}"

            item = {
                "text": final_text,
                "metadata": {
                    "source": clean_source,
                    "page_number": page_number,
                    "element_type": category,
                    "is_table": is_table,
                    "element_id": getattr(elem, 'id', None),
                },
            }
            processed_data.append(item)

        return processed_data

    @staticmethod
    def _html_to_markdown(html_content: str) -> str:
        """尝试使用 markdownify 库，如果不可用则回退到正则。"""
        try:
            import markdownify
            return markdownify.markdownify(html_content, heading_style="ATX")
        except ImportError:
            clean_text = re.sub('<[^<]+?>', '', html_content)
            return clean_text
        except Exception:
            return html_content
