import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
from tools import ArxivSearch
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoRA-ReadingAgent")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PaperSection:
    """论文章节类，用于表示论文的一个章节"""
    def __init__(self, title: str, content: str, level: int = 1):
        self.title = title
        self.content = content
        self.level = level  # 章节级别，1为最高级
        self.subsections = []  # 子章节

    def add_subsection(self, subsection: 'PaperSection'):
        """添加子章节"""
        self.subsections.append(subsection)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "subsections": [subsec.to_dict() for subsec in self.subsections]
        }

class Paper:
    """论文类，用于表示一篇完整的论文"""
    def __init__(self, paper_id: str, title: str, authors: List[str], abstract: str):
        self.paper_id = paper_id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.sections = []  # 论文章节
        self.full_text = ""  # 完整文本
        self.references = []  # 参考文献
        self.figures = []  # 图表
        self.tables = []  # 表格
        self.metadata = {}  # 元数据

    def add_section(self, section: PaperSection):
        """添加章节"""
        self.sections.append(section)

    def add_reference(self, reference: Dict[str, str]):
        """添加参考文献"""
        self.references.append(reference)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "sections": [section.to_dict() for section in self.sections],
            "references": self.references,
            "metadata": self.metadata
        }

class ReadingAgent:
    """
    专门用于阅读和理解论文的代理类
    """
    def __init__(self):
        """
        初始化ReadingAgent
        """
        self.arxiv_search = ArxivSearch()
        self.read_papers = {}  # 已读论文缓存
        logger.info("ReadingAgent初始化完成")

    def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        读取并解析论文
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            包含解析结果的字典
        """
        logger.info(f"ReadingAgent正在读取论文，ID: {paper_id}")
        
        # 检查缓存
        if paper_id in self.read_papers:
            logger.info(f"从缓存读取论文: {paper_id}")
            return {
                "status": "success",
                "message": "从缓存读取论文成功",
                "paper": self.read_papers[paper_id]
            }
        
        # 获取论文元数据
        metadata_result = self._get_metadata(paper_id)
        if metadata_result["status"] != "success":
            return metadata_result
        
        metadata = metadata_result["metadata"]
        
        # 获取论文全文
        fulltext_result = self._get_fulltext(paper_id)
        if fulltext_result["status"] != "success":
            return fulltext_result
        
        fulltext = fulltext_result["text"]
        
        # 创建论文对象
        paper = Paper(
            paper_id=paper_id,
            title=metadata.get("title", "Unknown Title"),
            authors=metadata.get("authors", []),
            abstract=metadata.get("summary", "No abstract available")
        )
        paper.metadata = metadata
        paper.full_text = fulltext
        
        # 解析论文结构
        self._parse_paper_structure(paper, fulltext)
        
        # 提取参考文献
        self._extract_references(paper, fulltext)
        
        # 缓存论文
        self.read_papers[paper_id] = paper
        
        return {
            "status": "success",
            "message": "论文读取和解析成功",
            "paper": paper
        }
    
    def _get_metadata(self, paper_id: str) -> Dict[str, Any]:
        """获取论文元数据"""
        try:
            metadata = self.arxiv_search.get_paper_metadata(paper_id)
            if not metadata:
                return {
                    "status": "error",
                    "message": "获取论文元数据失败",
                    "paper_id": paper_id,
                    "metadata": None
                }
            
            return {
                "status": "success",
                "message": "成功获取论文元数据",
                "paper_id": paper_id,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"获取元数据时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"获取元数据时出错: {str(e)}",
                "paper_id": paper_id,
                "metadata": None
            }
    
    def _get_fulltext(self, paper_id: str) -> Dict[str, Any]:
        """获取论文全文"""
        try:
            paper_text = self.arxiv_search.retrieve_full_paper_text(paper_id)
            
            if not paper_text or paper_text.startswith("ERROR") or paper_text == "EXTRACTION FAILED":
                return {
                    "status": "error",
                    "message": f"获取论文全文失败: {paper_text}",
                    "paper_id": paper_id,
                    "text": None
                }
            
            return {
                "status": "success",
                "message": "成功获取论文全文",
                "paper_id": paper_id,
                "text": paper_text
            }
        except Exception as e:
            logger.error(f"获取全文时出错: {str(e)}")
            return {
                "status": "error",
                "message": f"获取全文时出错: {str(e)}",
                "paper_id": paper_id,
                "text": None
            }
    
    def _parse_paper_structure(self, paper: Paper, fulltext: str) -> None:
        """
        解析论文结构
        
        参数:
            paper: 论文对象
            fulltext: 论文全文
        """
        # 分割成页面
        pages = fulltext.split("--- Page ")
        
        # 移除第一个空元素（如果存在）
        if pages and not pages[0].strip():
            pages = pages[1:]
        
        # 处理页眉页脚
        cleaned_pages = []
        for page in pages:
            if not page.strip():
                continue
                
            # 提取页码和内容
            page_parts = page.split("---\n", 1)
            if len(page_parts) > 1:
                content = page_parts[1]
                cleaned_pages.append(content)
            else:
                cleaned_pages.append(page)
        
        # 合并页面
        full_content = "\n".join(cleaned_pages)
        
        # 使用正则表达式识别章节标题
        section_pattern = r"(?:^|\n)(\d+(?:\.\d+)*)\s+([^\n]+)(?:\n|$)"
        matches = re.finditer(section_pattern, full_content)
        
        current_sections = {}  # 用于跟踪当前章节
        
        last_match_end = 0
        
        # 如果没有找到符合格式的章节，创建默认章节
        if not re.search(section_pattern, full_content):
            # 检测通用章节标题
            common_sections = ["Abstract", "Introduction", "Related Work", "Method", "Experiment", "Results", "Discussion", "Conclusion", "References"]
            for section_title in common_sections:
                pattern = fr"(?:^|\n)({section_title}(?:s)?)\s*(?:\n|$)"
                for match in re.finditer(pattern, full_content, re.IGNORECASE):
                    start_pos = match.start()
                    section_title = match.group(1)
                    
                    # 查找下一个章节的开始位置
                    next_section_match = None
                    for next_title in common_sections:
                        next_pattern = fr"(?:^|\n)({next_title}(?:s)?)\s*(?:\n|$)"
                        next_matches = list(re.finditer(next_pattern, full_content[start_pos + len(match.group(0)):], re.IGNORECASE))
                        if next_matches:
                            next_match = next_matches[0]
                            rel_pos = next_match.start()
                            abs_pos = start_pos + len(match.group(0)) + rel_pos
                            if next_section_match is None or abs_pos < next_section_match[0]:
                                next_section_match = (abs_pos, next_match.group(1))
                    
                    # 提取内容
                    end_pos = next_section_match[0] if next_section_match else len(full_content)
                    content = full_content[start_pos + len(match.group(0)):end_pos].strip()
                    
                    # 创建章节
                    section = PaperSection(title=section_title, content=content, level=1)
                    paper.add_section(section)
            
            # 如果仍然没有识别出章节，创建单个默认章节
            if not paper.sections:
                section = PaperSection(title="Content", content=full_content, level=1)
                paper.add_section(section)
            
            return
            
        # 处理识别出的结构化章节
        for match in matches:
            section_number = match.group(1)
            section_title = match.group(2)
            start_pos = match.end()
            
            # 计算章节级别
            level = len(section_number.split("."))
            
            # 如果有前一个匹配，获取内容
            if last_match_end > 0:
                content_start = last_match_end
                content_end = match.start()
                content = full_content[content_start:content_end].strip()
                
                # 检查当前级别和路径
                current_level = len(current_sections)
                
                # 创建章节对象
                section_obj = PaperSection(
                    title=current_sections.get(current_level, {}).get("title", ""),
                    content=content,
                    level=current_level
                )
                
                # 将章节添加到适当的位置
                if current_level == 1:
                    paper.add_section(section_obj)
                else:
                    parent_level = current_level - 1
                    parent_section = current_sections.get(parent_level, {}).get("section")
                    if parent_section:
                        parent_section.add_subsection(section_obj)
            
            # 更新当前章节信息
            current_sections[level] = {
                "title": section_title,
                "section": PaperSection(title=section_title, content="", level=level)
            }
            
            # 移除更高级别的章节
            keys_to_remove = [k for k in current_sections.keys() if k > level]
            for k in keys_to_remove:
                current_sections.pop(k, None)
            
            last_match_end = start_pos
        
        # 处理最后一个章节
        if last_match_end > 0:
            content = full_content[last_match_end:].strip()
            current_level = len(current_sections)
            
            section_obj = PaperSection(
                title=current_sections.get(current_level, {}).get("title", ""),
                content=content,
                level=current_level
            )
            
            if current_level == 1:
                paper.add_section(section_obj)
            else:
                parent_level = current_level - 1
                parent_section = current_sections.get(parent_level, {}).get("section")
                if parent_section:
                    parent_section.add_subsection(section_obj)
    
    def _extract_references(self, paper: Paper, fulltext: str) -> None:
        """
        提取参考文献
        
        参数:
            paper: 论文对象
            fulltext: 论文全文
        """
        # 查找参考文献部分
        references_section_patterns = [
            r"(?:^|\n)References\s*(?:\n|$)",
            r"(?:^|\n)REFERENCES\s*(?:\n|$)",
            r"(?:^|\n)Bibliography\s*(?:\n|$)",
            r"(?:^|\n)BIBLIOGRAPHY\s*(?:\n|$)"
        ]
        
        references_text = ""
        for pattern in references_section_patterns:
            matches = list(re.finditer(pattern, fulltext))
            if matches:
                start_pos = matches[-1].end()  # 使用最后一个匹配，因为可能在摘要中也提到"References"
                references_text = fulltext[start_pos:].strip()
                break
        
        if not references_text:
            logger.warning("未找到参考文献部分")
            return
        
        # 尝试识别参考文献格式
        # 格式1: [1] Author, Title...
        ref_pattern1 = r"\[\d+\]\s+([^\[\]]+)(?=\[\d+\]|$)"
        # 格式2: 1. Author, Title...
        ref_pattern2 = r"(?:^|\n)\d+\.\s+([^\n]+)"
        # 格式3: [Author, Year] Title...
        ref_pattern3 = r"\[([^\[\]]+)\]\s+([^\[\]]+)(?=\[|$)"
        
        references = []
        
        # 尝试格式1
        matches = list(re.finditer(ref_pattern1, references_text))
        if matches:
            for match in matches:
                ref_text = match.group(1).strip()
                references.append({"text": ref_text})
        
        # 如果格式1未找到匹配，尝试格式2
        if not references:
            matches = list(re.finditer(ref_pattern2, references_text))
            if matches:
                for match in matches:
                    ref_text = match.group(1).strip()
                    references.append({"text": ref_text})
        
        # 如果格式2未找到匹配，尝试格式3
        if not references:
            matches = list(re.finditer(ref_pattern3, references_text))
            if matches:
                for match in matches:
                    author_year = match.group(1).strip()
                    title = match.group(2).strip()
                    references.append({
                        "author_year": author_year,
                        "title": title
                    })
        
        # 如果仍未找到匹配，尝试按行分割
        if not references:
            lines = references_text.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Page") and len(line) > 20:  # 简单过滤无效行
                    references.append({"text": line})
        
        paper.references = references
    
    def summarize_paper(self, paper_id: str, max_length: int = 500) -> Dict[str, Any]:
        """
        生成论文摘要
        
        参数:
            paper_id: arXiv论文ID
            max_length: 摘要最大长度
            
        返回:
            包含摘要的字典
        """
        logger.info(f"ReadingAgent正在生成论文摘要，ID: {paper_id}")
        
        # 读取论文
        result = self.read_paper(paper_id)
        if result["status"] != "success":
            return {
                "status": "error",
                "message": f"无法获取论文: {result['message']}",
                "summary": None
            }
        
        paper = result["paper"]
        
        # 提取关键章节内容
        key_sections = ["abstract", "introduction", "conclusion"]
        section_texts = []
        
        # 添加摘要
        section_texts.append(paper.abstract)
        
        # 添加关键章节
        for section in paper.sections:
            if any(key in section.title.lower() for key in key_sections):
                section_texts.append(section.content)
        
        # 如果没有足够的内容，使用前几个章节
        if len(section_texts) < 2 and paper.sections:
            for section in paper.sections[:2]:
                if section.content and section.content not in section_texts:
                    section_texts.append(section.content)
        
        # 合并文本
        combined_text = " ".join(section_texts)
        
        # 分割成句子
        sentences = sent_tokenize(combined_text)
        
        # 简单的摘要生成：选择前几个句子
        num_sentences = min(10, len(sentences))
        summary_sentences = sentences[:num_sentences]
        
        # 组合成摘要
        summary = " ".join(summary_sentences)
        
        # 截断到最大长度
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(" ", 1)[0] + "..."
        
        return {
            "status": "success",
            "message": "论文摘要生成成功",
            "paper_id": paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "summary": summary
        }
    
    def extract_key_findings(self, paper_id: str) -> Dict[str, Any]:
        """
        提取论文的关键发现
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            包含关键发现的字典
        """
        logger.info(f"ReadingAgent正在提取论文关键发现，ID: {paper_id}")
        
        # 读取论文
        result = self.read_paper(paper_id)
        if result["status"] != "success":
            return {
                "status": "error",
                "message": f"无法获取论文: {result['message']}",
                "findings": None
            }
        
        paper = result["paper"]
        
        # 查找结果和讨论章节
        target_sections = []
        for section in paper.sections:
            if any(key in section.title.lower() for key in ["result", "finding", "discussion", "conclusion", "eval"]):
                target_sections.append(section)
        
        # 如果没有找到相关章节，使用所有章节
        if not target_sections and paper.sections:
            target_sections = paper.sections
        
        # 提取可能包含关键发现的句子
        finding_indicators = [
            "we find", "we found", "finding", "result", "shows that", "demonstrate", "suggest",
            "conclude", "conclusion", "evidence", "performance", "improvement", "better than",
            "outperforms", "state-of-the-art", "sota"
        ]
        
        findings = []
        
        for section in target_sections:
            sentences = sent_tokenize(section.content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # 忽略太短的句子
                    lowercase_sentence = sentence.lower()
                    if any(indicator in lowercase_sentence for indicator in finding_indicators):
                        findings.append({
                            "section": section.title,
                            "finding": sentence
                        })
        
        # 限制结果数量
        if len(findings) > 10:
            findings = findings[:10]
        
        # 如果没有找到明确的发现，返回最后一个章节的内容
        if not findings and paper.sections:
            last_section = paper.sections[-1]
            sentences = sent_tokenize(last_section.content)
            selected_sentences = sentences[:3] if len(sentences) > 3 else sentences
            
            findings = [{
                "section": last_section.title,
                "finding": sentence.strip()
            } for sentence in selected_sentences if len(sentence.strip()) > 20]
        
        return {
            "status": "success",
            "message": "论文关键发现提取成功",
            "paper_id": paper_id,
            "title": paper.title,
            "findings": findings
        }
    
    def compare_papers(self, paper_ids: List[str]) -> Dict[str, Any]:
        """
        比较多篇论文
        
        参数:
            paper_ids: arXiv论文ID列表
            
        返回:
            包含比较结果的字典
        """
        logger.info(f"ReadingAgent正在比较论文，IDs: {paper_ids}")
        
        if len(paper_ids) < 2:
            return {
                "status": "error",
                "message": "需要至少两篇论文进行比较",
                "comparison": None
            }
        
        papers = []
        
        # 读取所有论文
        for paper_id in paper_ids:
            result = self.read_paper(paper_id)
            if result["status"] == "success":
                papers.append(result["paper"])
            else:
                logger.warning(f"无法获取论文 {paper_id}: {result['message']}")
        
        if len(papers) < 2:
            return {
                "status": "error",
                "message": "无法获取足够的论文进行比较",
                "comparison": None
            }
        
        # 比较论文
        comparison = {
            "papers": [{
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "publication_date": paper.metadata.get("published", "Unknown")
            } for paper in papers],
            "common_authors": self._find_common_authors(papers),
            "topic_similarity": self._calculate_topic_similarity(papers),
            "key_differences": self._identify_key_differences(papers)
        }
        
        return {
            "status": "success",
            "message": "论文比较成功",
            "comparison": comparison
        }
    
    def _find_common_authors(self, papers: List[Paper]) -> List[str]:
        """查找共同作者"""
        if not papers:
            return []
            
        # 获取每篇论文的作者集合
        author_sets = [set(paper.authors) for paper in papers]
        
        # 找到交集
        common_authors = author_sets[0]
        for author_set in author_sets[1:]:
            common_authors = common_authors.intersection(author_set)
        
        return list(common_authors)
    
    def _calculate_topic_similarity(self, papers: List[Paper]) -> float:
        """计算主题相似度（简化版）"""
        if len(papers) < 2:
            return 0.0
            
        # 简单地比较摘要中的共同词汇
        abstracts = [paper.abstract.lower() for paper in papers]
        
        # 提取单词
        word_sets = [set(re.findall(r'\b\w+\b', abstract)) for abstract in abstracts]
        
        # 计算共同词汇比例
        common_words = word_sets[0]
        for word_set in word_sets[1:]:
            common_words = common_words.intersection(word_set)
        
        total_words = set()
        for word_set in word_sets:
            total_words = total_words.union(word_set)
        
        # 计算Jaccard相似度
        similarity = len(common_words) / len(total_words) if total_words else 0.0
        
        return similarity
    
    def _identify_key_differences(self, papers: List[Paper]) -> List[Dict[str, str]]:
        """识别关键差异（简化版）"""
        if len(papers) < 2:
            return []
            
        differences = []
        
        # 比较摘要中的不同点
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers):
                if i < j:  # 避免重复比较
                    # 提取独特词汇
                    words1 = set(re.findall(r'\b\w+\b', paper1.abstract.lower()))
                    words2 = set(re.findall(r'\b\w+\b', paper2.abstract.lower()))
                    
                    unique_words1 = words1 - words2
                    unique_words2 = words2 - words1
                    
                    # 构建差异描述
                    diff = {
                        "paper1_id": paper1.paper_id,
                        "paper2_id": paper2.paper_id,
                        "difference": f"Paper {paper1.paper_id} focuses more on {', '.join(list(unique_words1)[:5])}, " +
                                      f"while paper {paper2.paper_id} emphasizes {', '.join(list(unique_words2)[:5])}."
                    }
                    
                    differences.append(diff)
        
        return differences
    
    def create_literature_review(self, paper_ids: List[str]) -> Dict[str, Any]:
        """
        创建文献综述
        
        参数:
            paper_ids: arXiv论文ID列表
            
        返回:
            包含文献综述的字典
        """
        logger.info(f"ReadingAgent正在创建文献综述，共 {len(paper_ids)} 篇论文")
        
        if not paper_ids:
            return {
                "status": "error",
                "message": "需要至少一篇论文创建文献综述",
                "review": None
            }
        
        papers = []
        
        # 读取所有论文
        for paper_id in paper_ids:
            result = self.read_paper(paper_id)
            if result["status"] == "success":
                papers.append(result["paper"])
            else:
                logger.warning(f"无法获取论文 {paper_id}: {result['message']}")
        
        if not papers:
            return {
                "status": "error",
                "message": "无法获取任何论文",
                "review": None
            }
        
        # 按发布日期排序
        papers.sort(key=lambda p: p.metadata.get("published", ""), reverse=True)
        
        # 生成综述
        review = {
            "title": f"Literature Review: {papers[0].title.split(':')[0] if ':' in papers[0].title else papers[0].title}",
            "introduction": self._generate_review_introduction(papers),
            "paper_summaries": [self._generate_paper_summary(paper) for paper in papers],
            "key_findings": self._extract_common_findings(papers),
            "conclusion": self._generate_review_conclusion(papers)
        }
        
        return {
            "status": "success",
            "message": "文献综述创建成功",
            "review": review
        }
    
    def _generate_review_introduction(self, papers: List[Paper]) -> str:
        """生成综述介绍"""
        # 提取研究领域
        first_paper = papers[0]
        field = first_paper.title.split(':')[0] if ':' in first_paper.title else "this field"
        
        # 生成介绍
        introduction = f"This literature review examines {len(papers)} papers in the field of {field}. "
        introduction += f"The papers were published between {papers[-1].metadata.get('published', 'N/A')[:4]} and {papers[0].metadata.get('published', 'N/A')[:4]}. "
        
        # 添加主要作者
        all_authors = []
        for paper in papers:
            all_authors.extend(paper.authors)
        
        # 计算作者频率
        author_frequency = {}
        for author in all_authors:
            author_frequency[author] = author_frequency.get(author, 0) + 1
        
        # 找出出现次数最多的作者
        top_authors = sorted(author_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_authors:
            introduction += f"Key contributors in this area include {', '.join([author for author, _ in top_authors])}. "
        
        # 添加研究主题描述
        common_words = self._extract_common_keywords(papers)
        if common_words:
            introduction += f"The main research themes include {', '.join(common_words)}."
        
        return introduction
    
    def _generate_paper_summary(self, paper: Paper) -> Dict[str, str]:
        """为单篇论文生成摘要"""
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": ", ".join(paper.authors),
            "published": paper.metadata.get("published", "N/A"),
            "summary": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
            "key_points": self._extract_paper_key_points(paper)
        }
    
    def _extract_paper_key_points(self, paper: Paper) -> List[str]:
        """提取论文关键点"""
        key_points = []
        
        # 从摘要和结论中提取
        text = paper.abstract
        for section in paper.sections:
            if "conclusion" in section.title.lower():
                text += " " + section.content
        
        # 查找关键短语
        key_phrases = [
            "we propose", "we present", "we introduce", "we develop",
            "results show", "we find", "we demonstrate", "we achieve",
            "contribution", "outperforms", "novel", "state-of-the-art"
        ]
        
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # 忽略太短的句子
                lowercase_sentence = sentence.lower()
                if any(phrase in lowercase_sentence for phrase in key_phrases):
                    key_points.append(sentence)
        
        # 限制结果数量
        if len(key_points) > 3:
            key_points = key_points[:3]
        
        # 如果没有找到关键点，使用摘要的前几句话
        if not key_points and sentences:
            key_points = [sentences[0]]
            if len(sentences) > 1:
                key_points.append(sentences[1])
        
        return key_points
    
    def _extract_common_findings(self, papers: List[Paper]) -> List[str]:
        """提取共同发现"""
        # 提取每篇论文的关键点
        all_key_points = []
        for paper in papers:
            key_points = self._extract_paper_key_points(paper)
            all_key_points.extend(key_points)
        
        # 返回所有关键点（在实际应用中可以进一步归纳）
        return all_key_points[:5]  # 限制数量
    
    def _generate_review_conclusion(self, papers: List[Paper]) -> str:
        """生成综述结论"""
        # 简单结论
        conclusion = f"This literature review has examined {len(papers)} papers in the field. "
        
        # 添加研究趋势
        conclusion += "The research in this field is evolving, with recent papers focusing on "
        
        # 从最近的论文中提取主题
        recent_papers = papers[:3]  # 最近的三篇论文
        recent_topics = self._extract_common_keywords(recent_papers)
        
        if recent_topics:
            conclusion += ", ".join(recent_topics) + ". "
        else:
            conclusion += "various innovative approaches. "
        
        # 添加未来方向
        conclusion += "Future research could explore these areas in more depth and address remaining challenges."
        
        return conclusion
    
    def _extract_common_keywords(self, papers: List[Paper]) -> List[str]:
        """提取共同关键词"""
        # 合并所有摘要
        all_text = " ".join([paper.abstract for paper in papers])
        
        # 提取单词
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # 过滤停用词和短词
        stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are", "we"]
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # 计算词频
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 返回高频词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:10] if freq > 1]
        
        return top_words


# 测试代码
if __name__ == "__main__":
    # 创建代理
    agent = ReadingAgent()
    
    # 测试读取论文
    paper_id = "2304.01373"  # 示例论文ID
    result = agent.read_paper(paper_id)
    
    if result["status"] == "success":
        paper = result["paper"]
        print(f"成功读取论文: {paper.title}")
        print(f"章节数量: {len(paper.sections)}")
        
        # 生成摘要
        summary_result = agent.summarize_paper(paper_id)
        if summary_result["status"] == "success":
            print("\n论文摘要:")
            print(summary_result["summary"])
        
        # 提取关键发现
        findings_result = agent.extract_key_findings(paper_id)
        if findings_result["status"] == "success":
            print("\n关键发现:")
            for finding in findings_result["findings"]:
                print(f"- [{finding['section']}] {finding['finding']}")
    else:
        print(f"读取论文失败: {result['message']}") 