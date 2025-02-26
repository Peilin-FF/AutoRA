import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from tools import ArxivSearch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoRA-WebAgent")

class ArxivAgent:
    """
    专门用于处理arXiv论文的代理类
    """
    def __init__(self):
        """
        初始化ArxivAgent
        """
        self.arxiv_search = ArxivSearch()
        self.search_history = []
        logger.info("ArxivAgent初始化完成")
        
    def search_papers(self, query: str, max_results: int = 20) -> Dict[str, Any]:
        """
        搜索arXiv论文
        
        参数:
            query: 搜索查询字符串
            max_results: 返回的最大结果数
            
        返回:
            包含搜索结果的字典
        """
        logger.info(f"ArxivAgent正在搜索论文，查询: {query}, 最大结果数: {max_results}")
        
        # 记录搜索历史
        self.search_history.append(query)
        
        # 执行搜索
        papers_text = self.arxiv_search.find_papers_by_str(query, N=max_results)
        
        if not papers_text:
            return {
                "status": "error",
                "message": "搜索失败或未找到结果",
                "papers": []
            }
        
        # 解析结果
        papers = self._parse_papers_text(papers_text)
        
        return {
            "status": "success",
            "message": f"找到 {len(papers)} 篇相关论文",
            "query": query,
            "papers": papers
        }
    
    def _parse_papers_text(self, papers_text: str) -> List[Dict[str, str]]:
        """
        解析论文文本，转换为结构化数据
        
        参数:
            papers_text: 论文文本字符串
            
        返回:
            论文列表，每篇论文为一个字典
        """
        papers = []
        
        # 按论文分割文本
        paper_blocks = papers_text.split("\n\n")
        
        for block in paper_blocks:
            if not block.strip():
                continue
                
            paper_dict = {}
            lines = block.strip().split("\n")
            
            for line in lines:
                if not line or ":" not in line:
                    continue
                    
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "Title":
                    paper_dict["title"] = value
                elif key == "Authors":
                    paper_dict["authors"] = value
                elif key == "Summary":
                    paper_dict["summary"] = value
                elif key == "Publication Date":
                    paper_dict["publication_date"] = value
                elif key == "Categories":
                    paper_dict["categories"] = value
                elif key == "arXiv paper ID":
                    paper_dict["paper_id"] = value
            
            if paper_dict:  # 确保不添加空字典
                papers.append(paper_dict)
        
        return papers
    
    def get_paper_full_text(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文全文
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            包含论文全文的字典
        """
        logger.info(f"ArxivAgent正在获取论文全文，ID: {paper_id}")
        
        # 获取论文全文
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
    
    def get_paper_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        获取论文元数据
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            包含论文元数据的字典
        """
        logger.info(f"ArxivAgent正在获取论文元数据，ID: {paper_id}")
        
        # 获取论文元数据
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
    
    def analyze_papers(self, papers: List[Dict[str, str]], analysis_type: str = "trend") -> Dict[str, Any]:
        """
        分析论文集合
        
        参数:
            papers: 论文列表
            analysis_type: 分析类型，可选值: "trend", "topic", "author"
            
        返回:
            包含分析结果的字典
        """
        logger.info(f"ArxivAgent正在分析论文，分析类型: {analysis_type}, 论文数量: {len(papers)}")
        
        if not papers:
            return {
                "status": "error",
                "message": "没有论文可供分析",
                "analysis_type": analysis_type,
                "results": None
            }
        
        results = {}
        
        if analysis_type == "trend":
            # 按发布日期分析趋势
            date_counts = {}
            for paper in papers:
                date = paper.get("publication_date", "Unknown")
                if date in date_counts:
                    date_counts[date] += 1
                else:
                    date_counts[date] = 1
            
            # 按日期排序
            sorted_dates = sorted(date_counts.keys())
            results["date_distribution"] = {date: date_counts[date] for date in sorted_dates}
            
        elif analysis_type == "topic":
            # 按类别分析主题
            category_counts = {}
            for paper in papers:
                categories = paper.get("categories", "").split()
                for category in categories:
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1
            
            # 按频率排序
            sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            results["category_distribution"] = {cat: count for cat, count in sorted_categories}
            
        elif analysis_type == "author":
            # 按作者分析
            author_counts = {}
            for paper in papers:
                authors = paper.get("authors", "").split(", ")
                for author in authors:
                    if author in author_counts:
                        author_counts[author] += 1
                    else:
                        author_counts[author] = 1
            
            # 按频率排序
            sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
            results["author_distribution"] = {author: count for author, count in sorted_authors[:20]}  # 只取前20名
        
        return {
            "status": "success",
            "message": f"成功完成{analysis_type}分析",
            "analysis_type": analysis_type,
            "results": results
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> Dict[str, Any]:
        """
        保存结果到文件
        
        参数:
            results: 结果字典
            output_file: 输出文件路径
            
        返回:
            操作状态
        """
        logger.info(f"ArxivAgent正在保存结果到文件: {output_file}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
            # 保存为JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return {
                "status": "success",
                "message": f"结果已保存到 {output_file}",
                "file_path": output_file
            }
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            return {
                "status": "error",
                "message": f"保存结果失败: {str(e)}",
                "file_path": output_file
            }
    
    def get_search_history(self) -> List[str]:
        """
        获取搜索历史
        
        返回:
            搜索历史列表
        """
        return self.search_history


# 测试代码
if __name__ == "__main__":
    # 创建代理
    agent = ArxivAgent()
    
    # 测试搜索论文
    search_results = agent.search_papers("large language models", max_results=5)
    print(json.dumps(search_results, indent=2, ensure_ascii=False))
    
    # 如果搜索成功，测试获取论文全文
    if search_results["status"] == "success" and search_results["papers"]:
        paper_id = search_results["papers"][0]["paper_id"]
        
        # 获取元数据
        metadata_results = agent.get_paper_metadata(paper_id)
        print(json.dumps(metadata_results, indent=2, ensure_ascii=False))
        
        # 分析论文
        analysis_results = agent.analyze_papers(search_results["papers"], analysis_type="topic")
        print(json.dumps(analysis_results, indent=2, ensure_ascii=False))
        
        # 保存结果
        save_results = agent.save_results(search_results, "arxiv_search_results.json")
        print(json.dumps(save_results, indent=2, ensure_ascii=False)) 