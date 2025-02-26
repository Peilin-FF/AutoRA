#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import argparse
import time
import requests
from typing import List, Dict, Any
from agents import ArxivAgent
from reading_agents import ReadingAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MultimodalRAG-Demo")

class MultimodalRAGAnalyzer:
    """
    多模态RAG论文分析器
    用于查找、阅读和分析多模态RAG相关论文
    """
    def __init__(self):
        self.arxiv_agent = ArxivAgent()
        self.reading_agent = ReadingAgent()
        logger.info("多模态RAG分析器初始化完成")
    
    def search_papers(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        搜索相关论文
        
        参数:
            query: 搜索查询
            max_results: 最大结果数量
            
        返回:
            搜索结果
        """
        logger.info(f"搜索论文：{query}, 最大结果数：{max_results}")
        return self.arxiv_agent.search_papers(query, max_results=max_results)
    
    def get_citation_count(self, paper_id: str) -> int:
        """
        获取论文引用量
        使用Semantic Scholar API
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            引用数量
        """
        try:
            url = f"https://api.semanticscholar.org/v1/paper/arXiv:{paper_id}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                return data.get("citationCount", 0)
            else:
                logger.warning(f"获取论文引用量失败: {response.status_code}")
                return 0
        except Exception as e:
            logger.error(f"获取引用量时出错: {str(e)}")
            return 0
    
    def analyze_paper(self, paper_id: str, paper_title: str, paper_abstract: str) -> Dict[str, Any]:
        """
        分析单篇论文，提取关键信息
        
        参数:
            paper_id: 论文ID
            paper_title: 论文标题
            paper_abstract: 论文摘要
            
        返回:
            分析结果
        """
        logger.info(f"分析论文：{paper_title}")
        
        # 创建一个临时Paper对象让ReadingAgent处理
        paper = {
            "paper_id": paper_id,
            "title": paper_title,
            "abstract": paper_abstract
        }
        
        # 获取引用量
        citation_count = self.get_citation_count(paper_id)
        
        # 提取关键信息
        modalities = self._extract_modalities(paper_abstract)
        database_construction = self._extract_database_construction(paper_abstract)
        experiments = self._extract_experiments(paper_abstract)
        
        return {
            "paper_id": paper_id,
            "title": paper_title,
            "link": f"https://arxiv.org/abs/{paper_id}",
            "citation_count": citation_count,
            "abstract": paper_abstract,
            "modalities": modalities,
            "database_construction": database_construction,
            "experiments": experiments
        }
    
    def _extract_modalities(self, abstract: str) -> List[str]:
        """提取论文中涉及的模态"""
        modalities = []
        
        # 定义可能的模态关键词
        modality_keywords = {
            "text": ["text", "textual", "language", "linguistic", "word", "document", "sentence", "paragraph", "article", "nlp"],
            "image": ["image", "visual", "picture", "photo", "imagery", "vision", "pixel", "vqa", "cv"],
            "video": ["video", "clip", "frame", "footage", "movie", "film", "temporal"],
            "audio": ["audio", "sound", "speech", "voice", "acoustic", "auditory", "asr"],
            "code": ["code", "programming", "source code", "software", "program", "coding", "repository"],
            "graph": ["graph", "network", "knowledge graph", "kg", "tree", "ontology", "taxonomy"],
            "tabular": ["table", "tabular", "spreadsheet", "row", "column", "cell", "database", "structured data"]
        }
        
        # 检测模态
        abstract_lower = abstract.lower()
        
        for modality, keywords in modality_keywords.items():
            for keyword in keywords:
                if keyword in abstract_lower:
                    modalities.append(modality)
                    break
        
        # 如果没有找到特定模态，查找通用的"多模态"关键词
        if not modalities:
            multimodal_keywords = ["multimodal", "multi-modal", "multiple modalities", "cross-modal"]
            if any(keyword in abstract_lower for keyword in multimodal_keywords):
                modalities.append("multimodal (unspecified)")
        
        # 去重
        return list(set(modalities))
    
    def _extract_database_construction(self, abstract: str) -> str:
        """提取数据库构建方式"""
        # 数据库构建相关的关键句
        db_keywords = ["database", "index", "vector", "embedding", "store", "retrieval", "collection", 
                      "corpus", "knowledge base", "repository", "faiss", "milvus", "elasticsearch", 
                      "database construction", "index construction", "build", "chunk", "split", "encoder"]
        
        # 获取包含关键词的句子
        construction_info = []
        sentences = abstract.split(". ")
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in db_keywords):
                construction_info.append(sentence)
        
        # 如果没有找到明确的句子，返回未知
        if not construction_info:
            return "未在摘要中明确提及数据库构建方式"
        
        return ". ".join(construction_info) + "."
    
    def _extract_experiments(self, abstract: str) -> str:
        """提取实验相关信息"""
        # 实验相关的关键句
        exp_keywords = ["experiment", "evaluation", "benchmark", "performance", "result", "score", 
                       "accuracy", "precision", "recall", "f1", "metric", "outperform", "state-of-the-art",
                       "dataset", "test", "baseline", "comparison", "sota", "effectiveness"]
        
        # 获取包含关键词的句子
        experiment_info = []
        sentences = abstract.split(". ")
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in exp_keywords):
                experiment_info.append(sentence)
        
        # 如果没有找到明确的句子，返回未知
        if not experiment_info:
            return "未在摘要中明确提及实验细节"
        
        return ". ".join(experiment_info) + "."
    
    def run_analysis(self, query: str, max_results: int = 20, output_file: str = "multimodal_rag_analysis.json") -> Dict[str, Any]:
        """
        运行完整分析流程
        
        参数:
            query: 搜索查询
            max_results: 最大结果数量
            output_file: 输出文件名
            
        返回:
            分析报告
        """
        logger.info(f"开始运行多模态RAG分析流程，查询：{query}")
        
        # 1. 搜索论文
        search_results = self.search_papers(query, max_results=max_results)
        
        if search_results["status"] != "success" or not search_results["papers"]:
            logger.error("搜索论文失败或未找到结果")
            return {
                "status": "error",
                "message": "搜索论文失败或未找到结果",
                "analysis": None
            }
        
        papers = search_results["papers"]
        logger.info(f"找到 {len(papers)} 篇论文")
        
        # 2. 分析每篇论文
        analysis_results = []
        
        for paper in papers:
            paper_id = paper.get("paper_id", "")
            paper_title = paper.get("title", "")
            paper_abstract = paper.get("summary", "")
            
            if paper_id and paper_title and paper_abstract:
                analysis = self.analyze_paper(paper_id, paper_title, paper_abstract)
                analysis_results.append(analysis)
                time.sleep(1)  # 避免API限制
        
        # 3. 按引用量排序
        sorted_results = sorted(analysis_results, key=lambda x: x["citation_count"], reverse=True)
        
        # 4. 生成报告
        report = {
            "status": "success",
            "message": f"成功分析 {len(sorted_results)} 篇多模态RAG相关论文",
            "query": query,
            "papers": sorted_results
        }
        
        # 5. 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"分析结果已保存到 {output_file}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
        
        return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态RAG论文分析工具')
    parser.add_argument('--query', type=str, default="multimodal RAG retrieval augmented generation",
                        help='搜索查询 (默认: "multimodal RAG retrieval augmented generation")')
    parser.add_argument('--max-results', type=int, default=20,
                        help='最大结果数量 (默认: 20)')
    parser.add_argument('--output', type=str, default="multimodal_rag_analysis.json",
                        help='输出文件名 (默认: multimodal_rag_analysis.json)')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = MultimodalRAGAnalyzer()
    report = analyzer.run_analysis(args.query, args.max_results, args.output)
    
    # 打印简短摘要
    if report["status"] == "success":
        print("\n=== 多模态RAG论文分析摘要 ===")
        print(f"查询: {report['query']}")
        print(f"分析论文数: {len(report['papers'])}")
        print(f"结果已保存到: {args.output}")
        
        print("\n引用量排名前5的论文:")
        for i, paper in enumerate(report['papers'][:5]):
            print(f"{i+1}. {paper['title']}")
            print(f"   引用量: {paper['citation_count']}")
            print(f"   模态: {', '.join(paper['modalities'])}")
            print(f"   链接: {paper['link']}")
            print()
    else:
        print(f"分析失败: {report['message']}")

if __name__ == "__main__":
    main() 