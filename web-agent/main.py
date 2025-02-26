import os
import sys
import json
import argparse
import logging
from agents import ArxivAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AutoRA-WebAgent-Main")

def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description='AutoRA Web Agent - arXiv论文搜索和分析工具')
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 搜索论文命令
    search_parser = subparsers.add_parser('search', help='搜索arXiv论文')
    search_parser.add_argument('query', type=str, help='搜索查询字符串')
    search_parser.add_argument('--max-results', '-n', type=int, default=20, help='返回的最大结果数 (默认: 10)')
    search_parser.add_argument('--output', '-o', type=str, help='输出结果的JSON文件路径')
    
    # 获取论文全文命令
    fulltext_parser = subparsers.add_parser('fulltext', help='获取论文全文')
    fulltext_parser.add_argument('paper_id', type=str, help='arXiv论文ID')
    fulltext_parser.add_argument('--output', '-o', type=str, help='输出结果的文本文件路径')
    
    # 获取论文元数据命令
    metadata_parser = subparsers.add_parser('metadata', help='获取论文元数据')
    metadata_parser.add_argument('paper_id', type=str, help='arXiv论文ID')
    metadata_parser.add_argument('--output', '-o', type=str, help='输出结果的JSON文件路径')
    
    # 分析论文命令
    analyze_parser = subparsers.add_parser('analyze', help='分析论文集合')
    analyze_parser.add_argument('input_file', type=str, help='包含论文数据的JSON文件路径')
    analyze_parser.add_argument('--type', '-t', type=str, choices=['trend', 'topic', 'author'], default='trend', 
                               help='分析类型: trend(趋势), topic(主题), author(作者) (默认: trend)')
    analyze_parser.add_argument('--output', '-o', type=str, help='输出结果的JSON文件路径')
    
    return parser

def handle_search(args, agent):
    """处理搜索命令"""
    logger.info(f"执行搜索命令: {args.query}, 最大结果数: {args.max_results}")
    
    # 执行搜索
    results = agent.search_papers(args.query, max_results=args.max_results)
    
    # 打印结果摘要
    if results["status"] == "success":
        print(f"✅ 成功: {results['message']}")
        
        # 打印论文标题列表
        print("\n论文列表:")
        for i, paper in enumerate(results["papers"], 1):
            print(f"{i}. {paper.get('title', '无标题')} (ID: {paper.get('paper_id', 'unknown')})")
    else:
        print(f"❌ 错误: {results['message']}")
    
    # 如果指定了输出文件，保存结果
    if args.output:
        save_result = agent.save_results(results, args.output)
        if save_result["status"] == "success":
            print(f"\n结果已保存到: {save_result['file_path']}")
        else:
            print(f"\n❌ 保存结果失败: {save_result['message']}")
    
    return results

def handle_fulltext(args, agent):
    """处理获取论文全文命令"""
    logger.info(f"执行获取论文全文命令: {args.paper_id}")
    
    # 获取论文全文
    results = agent.get_paper_full_text(args.paper_id)
    
    # 打印结果摘要
    if results["status"] == "success":
        print(f"✅ 成功: {results['message']}")
        
        # 打印论文全文的前300个字符
        text_preview = results["text"][:300] + "..." if len(results["text"]) > 300 else results["text"]
        print(f"\n论文全文预览:\n{text_preview}")
    else:
        print(f"❌ 错误: {results['message']}")
    
    # 如果指定了输出文件，保存结果
    if args.output:
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            # 保存为文本文件
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(results["text"] if results["text"] else "获取论文全文失败")
            
            print(f"\n全文已保存到: {args.output}")
        except Exception as e:
            print(f"\n❌ 保存全文失败: {str(e)}")
    
    return results

def handle_metadata(args, agent):
    """处理获取论文元数据命令"""
    logger.info(f"执行获取论文元数据命令: {args.paper_id}")
    
    # 获取论文元数据
    results = agent.get_paper_metadata(args.paper_id)
    
    # 打印结果摘要
    if results["status"] == "success":
        print(f"✅ 成功: {results['message']}")
        
        # 打印元数据摘要
        metadata = results["metadata"]
        print("\n论文元数据:")
        print(f"标题: {metadata.get('title', '无标题')}")
        print(f"作者: {', '.join(metadata.get('authors', ['未知']))}")
        print(f"发布日期: {metadata.get('published', '未知')}")
        print(f"更新日期: {metadata.get('updated', '未知')}")
        print(f"分类: {' '.join(metadata.get('categories', ['未知']))}")
        print(f"DOI: {metadata.get('doi', '未知')}")
        print(f"PDF URL: {metadata.get('pdf_url', '未知')}")
    else:
        print(f"❌ 错误: {results['message']}")
    
    # 如果指定了输出文件，保存结果
    if args.output:
        save_result = agent.save_results(results, args.output)
        if save_result["status"] == "success":
            print(f"\n元数据已保存到: {save_result['file_path']}")
        else:
            print(f"\n❌ 保存元数据失败: {save_result['message']}")
    
    return results

def handle_analyze(args, agent):
    """处理分析论文命令"""
    logger.info(f"执行分析论文命令: {args.input_file}, 分析类型: {args.type}")
    
    # 读取输入文件
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取论文列表
        if "papers" in data:
            papers = data["papers"]
        else:
            papers = data  # 假设文件直接包含论文列表
        
        # 执行分析
        results = agent.analyze_papers(papers, analysis_type=args.type)
        
        # 打印结果摘要
        if results["status"] == "success":
            print(f"✅ 成功: {results['message']}")
            
            # 打印分析结果
            print(f"\n{args.type}分析结果:")
            if args.type == "trend":
                for date, count in results["results"].get("date_distribution", {}).items():
                    print(f"{date}: {count}篇论文")
            elif args.type == "topic":
                for category, count in list(results["results"].get("category_distribution", {}).items())[:10]:
                    print(f"{category}: {count}篇论文")
                if len(results["results"].get("category_distribution", {})) > 10:
                    print("...")
            elif args.type == "author":
                for author, count in list(results["results"].get("author_distribution", {}).items())[:10]:
                    print(f"{author}: {count}篇论文")
                if len(results["results"].get("author_distribution", {})) > 10:
                    print("...")
        else:
            print(f"❌ 错误: {results['message']}")
        
        # 如果指定了输出文件，保存结果
        if args.output:
            save_result = agent.save_results(results, args.output)
            if save_result["status"] == "success":
                print(f"\n分析结果已保存到: {save_result['file_path']}")
            else:
                print(f"\n❌ 保存分析结果失败: {save_result['message']}")
        
        return results
    except Exception as e:
        print(f"❌ 读取或分析输入文件失败: {str(e)}")
        return {"status": "error", "message": f"读取或分析输入文件失败: {str(e)}"}

def main():
    """主函数"""
    # 设置命令行参数解析
    parser = setup_argparse()
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return
    
    # 创建代理
    agent = ArxivAgent()
    
    # 根据命令执行相应的处理函数
    if args.command == 'search':
        handle_search(args, agent)
    elif args.command == 'fulltext':
        handle_fulltext(args, agent)
    elif args.command == 'metadata':
        handle_metadata(args, agent)
    elif args.command == 'analyze':
        handle_analyze(args, agent)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main() 