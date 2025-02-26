import os
import time
import arxiv
from semanticscholar import SemanticScholar
from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoRA-WebAgent")

class ArxivSearch:
    """
    用于从arXiv搜索和获取论文的类
    """
    def __init__(self):
        # 构造默认API客户端
        self.sch_engine = arxiv.Client()
        logger.info("ArxivSearch初始化完成")
        
    def _process_query(self, query: str) -> str:
        """
        处理查询字符串，使其适合在MAX_QUERY_LENGTH内保留尽可能多的信息
        """
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # 分割成单词
        words = query.split()
        processed_query = []
        current_length = 0
        
        # 在保持在限制范围内的同时添加单词
        # 考虑单词之间的空格
        for word in words:
            # +1 用于单词之间添加的空格
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break
            
        return ' '.join(processed_query)
    
    def find_papers_by_str(self, query, N=20):
        """
        通过字符串查询查找论文
        
        参数:
            query: 搜索查询字符串
            N: 返回的最大结果数
            
        返回:
            包含论文摘要的字符串
        """
        logger.info(f"正在搜索arXiv论文，查询: {query}, 最大结果数: {N}")
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results`是一个生成器；可以逐个迭代其元素
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Authors: {', '.join(author.name for author in r.authors)}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"Categories: {' '.join(r.categories)}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                
                logger.info(f"找到 {len(paper_sums)} 篇相关论文")
                time.sleep(2.0)  # 避免API限制
                return "\n".join(paper_sums)
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"搜索失败 (尝试 {retry_count}/{max_retries}): {str(e)}")
                if retry_count < max_retries:
                    # 递增延时
                    time.sleep(2 * retry_count)
                    continue
                
        logger.error("所有重试均失败")
        return None

    def retrieve_full_paper_text(self, query):
        """
        获取完整论文文本
        
        参数:
            query: arXiv论文ID
            
        返回:
            论文全文的字符串
        """
        logger.info(f"正在获取论文全文，ID: {query}")
        pdf_text = str()
        try:
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
            # 将PDF下载到当前工作目录，使用自定义文件名
            paper.download_pdf(filename="downloaded-paper.pdf")
            logger.info(f"已下载论文: {paper.title}")
            
            # 创建PDF阅读器对象
            reader = PdfReader('downloaded-paper.pdf')
            # 遍历所有页面
            for page_number, page in enumerate(reader.pages, start=1):
                # 从页面提取文本
                try:
                    text = page.extract_text()
                    # 处理文本
                    pdf_text += f"--- Page {page_number} ---\n"
                    pdf_text += text
                    pdf_text += "\n"
                except Exception as e:
                    logger.error(f"提取页面 {page_number} 文本失败: {str(e)}")
                    os.remove("downloaded-paper.pdf")
                    time.sleep(2.0)
                    return "EXTRACTION FAILED"

            os.remove("downloaded-paper.pdf")
            logger.info("已删除临时PDF文件")
            time.sleep(2.0)  # 避免API限制
            return pdf_text
        except Exception as e:
            logger.error(f"获取论文全文失败: {str(e)}")
            if os.path.exists("downloaded-paper.pdf"):
                os.remove("downloaded-paper.pdf")
            return f"ERROR: {str(e)}"

    def get_paper_metadata(self, paper_id):
        """
        获取论文的元数据
        
        参数:
            paper_id: arXiv论文ID
            
        返回:
            包含论文元数据的字典
        """
        try:
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
            metadata = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary,
                "published": str(paper.published),
                "updated": str(paper.updated),
                "categories": paper.categories,
                "doi": paper.doi,
                "pdf_url": paper.pdf_url,
                "entry_id": paper.entry_id
            }
            return metadata
        except Exception as e:
            logger.error(f"获取论文元数据失败: {str(e)}")
            return None


class SemanticScholarSearch:
    """
    用于从Semantic Scholar搜索和获取论文的类
    """
    def __init__(self):
        self.sch_engine = SemanticScholar(retry=False)
        logger.info("SemanticScholarSearch初始化完成")

    def find_papers_by_str(self, query, N=10, min_citation_count=3):
        """
        通过字符串查询查找论文
        
        参数:
            query: 搜索查询字符串
            N: 返回的最大结果数
            min_citation_count: 最小引用次数
            
        返回:
            包含论文摘要的列表
        """
        logger.info(f"正在搜索Semantic Scholar论文，查询: {query}, 最大结果数: {N}")
        try:
            paper_sums = list()
            results = self.sch_engine.search_paper(
                query, 
                limit=N, 
                min_citation_count=min_citation_count, 
                open_access_pdf=True
            )
            
            for _i in range(len(results)):
                try:
                    paper_sum = f'Title: {results[_i].title}\n'
                    paper_sum += f'Authors: {", ".join([author.name for author in results[_i].authors])}\n'
                    paper_sum += f'Abstract: {results[_i].abstract}\n'
                    paper_sum += f'Citations: {results[_i].citationCount}\n'
                    
                    if hasattr(results[_i], 'publicationDate') and results[_i].publicationDate:
                        paper_sum += f'Release Date: year {results[_i].publicationDate.year}, month {results[_i].publicationDate.month}, day {results[_i].publicationDate.day}\n'
                    
                    paper_sum += f'Venue: {results[_i].venue}\n'
                    
                    if hasattr(results[_i], 'externalIds') and results[_i].externalIds and 'DOI' in results[_i].externalIds:
                        paper_sum += f'Paper ID: {results[_i].externalIds["DOI"]}\n'
                    elif hasattr(results[_i], 'paperId'):
                        paper_sum += f'Paper ID: {results[_i].paperId}\n'
                        
                    paper_sums.append(paper_sum)
                except Exception as e:
                    logger.warning(f"处理论文 {_i} 时出错: {str(e)}")
                    continue
                    
            logger.info(f"找到 {len(paper_sums)} 篇相关论文")
            return paper_sums
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return []

    def get_paper_details(self, paper_id):
        """
        获取论文的详细信息
        
        参数:
            paper_id: Semantic Scholar论文ID或DOI
            
        返回:
            包含论文详细信息的字典
        """
        try:
            paper = self.sch_engine.get_paper(paper_id)
            return paper
        except Exception as e:
            logger.error(f"获取论文详细信息失败: {str(e)}")
            return None


class GoogleScholarSearch:
    """
    用于从Google Scholar搜索论文的类
    注意：Google Scholar没有官方API，此实现使用网页抓取，可能会被封IP
    """
    def __init__(self):
        self.base_url = "https://scholar.google.com/scholar"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        logger.info("GoogleScholarSearch初始化完成")
        
    def find_papers_by_str(self, query, N=10):
        """
        通过字符串查询查找论文
        
        参数:
            query: 搜索查询字符串
            N: 返回的最大结果数
            
        返回:
            包含论文摘要的列表
        """
        logger.info(f"正在搜索Google Scholar论文，查询: {query}, 最大结果数: {N}")
        try:
            params = {
                'q': query,
                'hl': 'en',
                'num': min(N, 20)  # Google Scholar通常限制每页结果数
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            articles = soup.find_all('div', class_='gs_ri')
            
            results = []
            for article in articles[:N]:
                try:
                    title_tag = article.find('h3', class_='gs_rt')
                    title = title_tag.text if title_tag else "No title found"
                    
                    authors_venue_tag = article.find('div', class_='gs_a')
                    authors_venue = authors_venue_tag.text if authors_venue_tag else "No authors/venue found"
                    
                    snippet_tag = article.find('div', class_='gs_rs')
                    snippet = snippet_tag.text if snippet_tag else "No snippet found"
                    
                    cite_tag = article.find('div', class_='gs_fl')
                    citations = "Unknown"
                    if cite_tag:
                        cite_links = cite_tag.find_all('a')
                        for link in cite_links:
                            if 'Cited by' in link.text:
                                citations = link.text.replace('Cited by', '').strip()
                                break
                    
                    paper_info = f"Title: {title}\n"
                    paper_info += f"Authors/Venue: {authors_venue}\n"
                    paper_info += f"Snippet: {snippet}\n"
                    paper_info += f"Citations: {citations}\n"
                    
                    results.append(paper_info)
                except Exception as e:
                    logger.warning(f"处理Google Scholar结果时出错: {str(e)}")
                    continue
            
            logger.info(f"找到 {len(results)} 篇相关论文")
            return results
            
        except Exception as e:
            logger.error(f"Google Scholar搜索失败: {str(e)}")
            return []


# 测试代码
if __name__ == "__main__":
    # 测试ArxivSearch
    arxiv_search = ArxivSearch()
    papers = arxiv_search.find_papers_by_str("large language models", N=3)
    print(papers)
    
    # 测试SemanticScholarSearch
    ss_search = SemanticScholarSearch()
    papers = ss_search.find_papers_by_str("large language models", N=3)
    print(papers) 