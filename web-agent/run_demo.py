#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from multimodal_rag_demo import MultimodalRAGAnalyzer
from discussion_agents import DiscussionManager

def main():
    """运行多模态RAG论文分析和讨论演示"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多模态RAG文献分析和讨论演示')
    parser.add_argument('--query', type=str, default="multimodal RAG retrieval augmented generation",
                       help='搜索查询 (默认: "multimodal RAG retrieval augmented generation")')
    parser.add_argument('--max-results', type=int, default=20,
                       help='最大结果数量 (默认: 20)')
    parser.add_argument('--output', type=str, default="multimodal_rag_analysis.json",
                       help='分析输出文件名 (默认: multimodal_rag_analysis.json)')
    parser.add_argument('--with-discussion', action='store_true',
                       help='是否启用论文讨论 (默认: False)')
    parser.add_argument('--discussion-rounds', type=int, default=2,
                       help='每个讨论主题的轮数 (默认: 2)')
    parser.add_argument('--discussion-dir', type=str, default="discussions",
                       help='讨论记录保存目录 (默认: discussions)')
    
    args = parser.parse_args()
    
    print("="*50)
    print("多模态RAG文献分析与讨论演示")
    print("="*50)
    
    # 设置查询和输出文件
    query = args.query
    chinese_query = "包含多种模态的RAG文献"  # 原始中文查询参考
    output_file = args.output
    max_results = args.max_results
    
    print(f"中文查询参考: {chinese_query}")
    print(f"实际查询: {query}")
    print(f"最大结果数: {max_results}")
    print(f"输出文件: {output_file}")
    if args.with_discussion:
        print(f"讨论功能: 已启用")
        print(f"每主题讨论轮数: {args.discussion_rounds}")
        print(f"讨论记录目录: {args.discussion_dir}")
    print("="*50)
    
    # 创建分析器并运行
    analyzer = MultimodalRAGAnalyzer()
    print("开始分析论文，请稍候...")
    report = analyzer.run_analysis(query, max_results, output_file)
    
    # 打印结果摘要
    if report["status"] == "success":
        print("\n分析成功完成!")
        print(f"共分析论文数: {len(report['papers'])}")
        
        print("\n引用量排名前5的论文:")
        top_papers = report['papers'][:5]
        for i, paper in enumerate(top_papers):
            print(f"\n{i+1}. {paper['title']}")
            print(f"    引用量: {paper['citation_count']}")
            print(f"    链接: {paper['link']}")
            print(f"    模态: {', '.join(paper['modalities'])}")
            print(f"    数据库构建: {paper['database_construction'][:150]}...")
            print(f"    实验: {paper['experiments'][:150]}...")
        
        # 如果启用了讨论功能，对引用量最高的论文进行讨论
        if args.with_discussion and top_papers:
            # 选择引用量最高的论文进行讨论
            top_paper = top_papers[0]
            paper_id = top_paper['paper_id']
            paper_title = top_paper['title']
            
            print("\n"+"="*50)
            print(f"开始对论文进行讨论: {paper_title}")
            print("="*50)
            
            # 创建讨论管理器
            discussion_manager = DiscussionManager(output_dir=args.discussion_dir)
            
            # 进行讨论
            print("开始学生-教师讨论，请稍候...")
            discussion_result = discussion_manager.conduct_discussion(
                paper_id=paper_id,
                rounds_per_topic=args.discussion_rounds
            )
            
            # 显示讨论结果
            if discussion_result["status"] == "success":
                print("\n讨论成功完成!")
                print(f"讨论主题数: {discussion_result['topics_count']}")
                print(f"每个主题轮数: {discussion_result['rounds_per_topic']}")
                print(f"总讨论轮数: {discussion_result['total_rounds']}")
                print(f"讨论记录保存在: {os.path.join(args.discussion_dir, f'discussion_{paper_id}.json')}")
            else:
                print(f"\n讨论失败: {discussion_result['message']}")
    else:
        print(f"\n分析失败: {report['message']}")
    
    print("\n"+"="*50)
    print(f"完整分析结果已保存到: {output_file}")
    print("您可以使用以下命令查看结果:")
    print(f"python -m json.tool {output_file} | more")
    if args.with_discussion:
        print(f"\n讨论记录保存在 {args.discussion_dir} 目录下")
        print("您可以使用以下命令查看讨论记录:")
        if top_papers:
            paper_id = top_papers[0]['paper_id']
            print(f"python -m json.tool {os.path.join(args.discussion_dir, f'discussion_{paper_id}.json')} | more")
    print("="*50)

if __name__ == "__main__":
    main() 