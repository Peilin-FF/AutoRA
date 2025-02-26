#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
import time
import random
from typing import List, Dict, Any, Optional
from reading_agents import ReadingAgent

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Discussion-Agents")

# 定义常见的讨论问题
DISCUSSION_TOPICS = [
    "这篇论文提出的多模态RAG方法有哪些创新点？",
    "论文中使用了哪些模态，它们是如何融合的？",
    "该论文的数据库索引构建方式有什么特点？",
    "论文中的实验设计如何验证了其方法的有效性？",
    "与现有工作相比，这种多模态RAG方法有哪些优势和局限性？",
    "论文中的检索增强生成过程是如何处理多模态数据的？",
    "这种方法在实际应用中可能面临哪些挑战？",
    "论文是否解决了多模态RAG中的跨模态对齐问题？如何解决的？"
]

class StudentAgent:
    """
    学生代理，负责阅读论文并回答问题
    """
    def __init__(self, reading_agent: ReadingAgent):
        self.reading_agent = reading_agent
        self.knowledge_level = random.uniform(0.6, 0.9)  # 随机知识水平，模拟不同学生
        logger.info(f"StudentAgent初始化完成，知识水平：{self.knowledge_level:.2f}")
    
    def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """读取论文"""
        logger.info(f"StudentAgent正在阅读论文：{paper_id}")
        return self.reading_agent.read_paper(paper_id)
    
    def answer_question(self, paper: Dict[str, Any], question: str, 
                        previous_feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        回答关于论文的问题
        
        参数:
            paper: 论文数据
            question: 问题
            previous_feedback: 之前的反馈（如果是改进回答）
            
        返回:
            答案和推理过程
        """
        logger.info(f"StudentAgent正在回答问题：{question}")
        
        # 模拟思考时间
        time.sleep(random.uniform(1, 2))
        
        paper_content = paper.get("paper", {})
        
        # 获取论文信息
        title = paper_content.get("title", "未知标题")
        abstract = paper_content.get("abstract", "未知摘要")
        
        # 提取可能相关的内容
        relevant_content = abstract
        
        # 如果有章节信息，尝试找到相关章节
        sections = paper_content.get("sections", [])
        for section in sections:
            section_title = section.get("title", "").lower()
            if any(keyword in section_title.lower() for keyword in ["method", "approach", "model", 
                                                                  "experiment", "result", "discussion"]):
                relevant_content += " " + section.get("content", "")
        
        # 制作答案（简化版，实际中可以使用更复杂的逻辑或调用LLM）
        confidence = self.knowledge_level * random.uniform(0.8, 1.0)
        
        # 如果有之前的反馈，模拟学习和改进
        improved = False
        if previous_feedback:
            confidence += random.uniform(0.05, 0.15)  # 提高置信度
            improved = True
        
        # 模拟基于问题内容生成答案
        if "模态" in question or "modality" in question.lower():
            answer = self._answer_about_modalities(abstract, confidence, improved)
        elif "数据库" in question or "database" in question.lower() or "索引" in question:
            answer = self._answer_about_database(abstract, confidence, improved)
        elif "实验" in question or "experiment" in question.lower():
            answer = self._answer_about_experiments(abstract, confidence, improved)
        elif "创新" in question or "novel" in question.lower() or "innovation" in question.lower():
            answer = self._answer_about_innovation(abstract, confidence, improved)
        else:
            answer = self._answer_general(abstract, confidence, improved)
        
        # 添加思考过程
        reasoning = f"我首先分析了论文的摘要和相关章节，提取了与问题'{question}'相关的关键信息。"
        if improved:
            reasoning += f" 根据老师之前的反馈'{previous_feedback.get('feedback', '')}'，我特别注意了{previous_feedback.get('suggestions', '')}。"
        
        return {
            "answer": answer,
            "reasoning": reasoning,
            "confidence": confidence
        }
    
    def _answer_about_modalities(self, content: str, confidence: float, improved: bool) -> str:
        """生成关于模态的回答"""
        modality_keywords = {
            "text": ["文本", "text", "textual", "language", "linguistic", "nlp"],
            "image": ["图像", "image", "visual", "picture", "vision", "cv"],
            "video": ["视频", "video", "clip", "frame", "footage"],
            "audio": ["音频", "audio", "sound", "speech", "voice"]
        }
        
        detected_modalities = []
        for modality, keywords in modality_keywords.items():
            if any(keyword in content.lower() for keyword in keywords):
                detected_modalities.append(modality)
        
        if not detected_modalities:
            detected_modalities = ["text"]  # 默认至少有文本模态
            
        fusion_methods = ["attention机制", "跨模态注意力", "多模态嵌入", "特征融合", "联合表示学习"]
        selected_fusion = random.sample(fusion_methods, min(2, len(fusion_methods)))
        
        quality = "详细" if confidence > 0.8 or improved else "基本"
        depth = "深入分析了各模态特点" if confidence > 0.85 or improved else "提到了基本用法"
        
        return f"这篇论文使用了{', '.join(detected_modalities)}等{len(detected_modalities)}种模态。论文{quality}描述了如何通过{', '.join(selected_fusion)}实现多模态信息的融合。{depth}，并将它们整合到RAG框架中，增强了检索和生成的效果。"
    
    def _answer_about_database(self, content: str, confidence: float, improved: bool) -> str:
        """生成关于数据库构建的回答"""
        db_approaches = ["向量数据库", "分层索引", "多模态索引", "跨模态检索", "FAISS", "Milvus"]
        selected_approaches = random.sample(db_approaches, min(2, len(db_approaches)))
        
        embedding_models = ["CLIP", "BERT", "RoBERTa", "Vision Transformer", "多模态编码器"]
        selected_models = random.sample(embedding_models, min(2, len(embedding_models)))
        
        detail_level = "详细" if confidence > 0.8 or improved else "简要"
        critique = "并分析了其优缺点" if confidence > 0.85 or improved else ""
        
        return f"论文采用了{', '.join(selected_approaches)}来构建多模态数据库。{detail_level}介绍了使用{', '.join(selected_models)}进行特征提取和索引构建的过程{critique}。数据库支持高效的跨模态检索，能够基于一种模态的查询检索出包含多种模态的相关内容。"
    
    def _answer_about_experiments(self, content: str, confidence: float, improved: bool) -> str:
        """生成关于实验的回答"""
        datasets = ["MS-COCO", "Flickr30k", "WebQA", "MMMU", "自建多模态数据集"]
        selected_datasets = random.sample(datasets, min(2, len(datasets)))
        
        metrics = ["准确率", "Recall@K", "BLEU", "ROUGE", "人工评估"]
        selected_metrics = random.sample(metrics, min(3, len(metrics)))
        
        baselines = ["传统RAG", "单模态方法", "之前的多模态方法", "大型语言模型"]
        selected_baselines = random.sample(baselines, min(2, len(baselines)))
        
        detail_level = "深入" if confidence > 0.8 or improved else "基本"
        analysis = "并进行了详细的消融实验分析不同组件的贡献" if confidence > 0.85 or improved else ""
        
        return f"论文在{', '.join(selected_datasets)}等数据集上进行了{detail_level}的实验评估{analysis}。使用{', '.join(selected_metrics)}等指标与{', '.join(selected_baselines)}进行了比较，结果表明所提出的多模态RAG方法取得了显著的性能提升。"
    
    def _answer_about_innovation(self, content: str, confidence: float, improved: bool) -> str:
        """生成关于创新点的回答"""
        innovations = [
            "新的多模态融合机制",
            "改进的跨模态检索算法",
            "高效的多模态索引结构",
            "端到端的多模态RAG框架",
            "适应性强的模态平衡方法"
        ]
        
        num_innovations = 3 if confidence > 0.8 or improved else 2
        selected_innovations = random.sample(innovations, num_innovations)
        
        depth = "详细分析" if confidence > 0.85 or improved else "介绍"
        
        return f"这篇论文的主要创新点包括：{'; '.join(selected_innovations)}。论文{depth}了这些创新如何解决现有多模态RAG系统中的挑战，特别是在处理模态间信息不平衡和跨模态语义对齐方面取得了进展。"
    
    def _answer_general(self, content: str, confidence: float, improved: bool) -> str:
        """生成通用回答"""
        quality = "非常全面" if confidence > 0.8 or improved else "基本"
        depth = "深入分析了关键技术和挑战" if confidence > 0.85 or improved else "介绍了基本方法"
        
        return f"这篇论文提出了一种{quality}的多模态RAG方法，能够处理多种模态的信息并增强生成效果。论文{depth}，包括多模态数据的处理、跨模态检索和融合生成等方面。实验结果表明该方法在多个基准测试中取得了良好的性能。"


class TeacherAgent:
    """
    教师代理，负责评价学生回答并提供指导
    """
    def __init__(self):
        self.reading_agent = ReadingAgent()
        logger.info("TeacherAgent初始化完成")
    
    def read_paper(self, paper_id: str) -> Dict[str, Any]:
        """读取论文"""
        logger.info(f"TeacherAgent正在阅读论文：{paper_id}")
        return self.reading_agent.read_paper(paper_id)
    
    def evaluate_answer(self, paper: Dict[str, Any], question: str, 
                         student_response: Dict[str, Any], round_num: int) -> Dict[str, Any]:
        """
        评价学生的回答
        
        参数:
            paper: 论文数据
            question: 问题
            student_response: 学生回答
            round_num: 讨论轮次
            
        返回:
            评价和建议
        """
        logger.info("TeacherAgent正在评价学生回答")
        
        # 模拟思考时间
        time.sleep(random.uniform(1, 1.5))
        
        student_answer = student_response.get("answer", "")
        student_confidence = student_response.get("confidence", 0.5)
        
        # 根据轮次调整评分基准
        base_score = 70 if round_num == 1 else 80
        
        # 评分计算
        content_score = self._evaluate_content(student_answer, question)
        reasoning_score = min(100, base_score + student_confidence * 20)
        
        # 最终评分
        final_score = (content_score * 0.7 + reasoning_score * 0.3)
        
        # 生成反馈
        feedback, suggestions = self._generate_feedback(student_answer, question, round_num, final_score)
        
        return {
            "feedback": feedback,
            "suggestions": suggestions,
            "score": round(final_score, 1),
            "content_score": round(content_score, 1),
            "reasoning_score": round(reasoning_score, 1)
        }
    
    def _evaluate_content(self, answer: str, question: str) -> float:
        """评估回答内容的质量"""
        # 检查答案长度
        length_score = min(100, max(50, len(answer) / 10))
        
        # 检查关键词覆盖率
        keywords = self._extract_question_keywords(question)
        covered_keywords = sum(1 for kw in keywords if kw.lower() in answer.lower())
        keyword_score = min(100, (covered_keywords / max(1, len(keywords))) * 100)
        
        # 检查答案结构
        structure_score = 85  # 默认较高的结构分
        if "。" not in answer or len(answer.split("。")) < 3:
            structure_score = 70  # 结构不够完善
        
        # 加权平均
        return length_score * 0.2 + keyword_score * 0.5 + structure_score * 0.3
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """从问题中提取关键词"""
        common_keywords = ["多模态", "RAG", "模态", "融合", "数据库", "索引", "实验", "创新", "方法", "挑战"]
        return [kw for kw in common_keywords if kw in question]
    
    def _generate_feedback(self, answer: str, question: str, round_num: int, score: float) -> tuple:
        """生成反馈和建议"""
        # 根据评分确定反馈类型
        if score >= 90:
            feedback_type = "非常好"
            tone = "赞赏"
        elif score >= 80:
            feedback_type = "良好"
            tone = "积极"
        elif score >= 70:
            feedback_type = "一般"
            tone = "中性"
        else:
            feedback_type = "需要改进"
            tone = "鼓励"
        
        # 根据问题类型生成具体反馈
        if "模态" in question:
            specific_feedback = self._feedback_for_modality(answer, tone)
        elif "数据库" in question or "索引" in question:
            specific_feedback = self._feedback_for_database(answer, tone)
        elif "实验" in question:
            specific_feedback = self._feedback_for_experiments(answer, tone)
        elif "创新" in question:
            specific_feedback = self._feedback_for_innovation(answer, tone)
        else:
            specific_feedback = self._feedback_general(answer, tone)
        
        # 根据轮次和评分生成建议
        if round_num == 1:
            if score < 75:
                suggestions = "请更详细地解释论文中的关键概念，并提供具体例子"
            elif score < 85:
                suggestions = "可以添加更多关于论文方法的技术细节，以及与现有工作的比较"
            else:
                suggestions = "尝试分析一下这种方法可能的局限性和未来改进方向"
        else:  # 第二轮或更多轮
            if score < 80:
                suggestions = "请重新组织你的回答结构，确保涵盖问题的所有方面"
            elif score < 90:
                suggestions = "可以更深入地讨论论文方法背后的原理和理论基础"
            else:
                suggestions = "你的回答已经很好，可以尝试从更广阔的背景角度进行讨论"
        
        feedback = f"{feedback_type}。{specific_feedback}"
        return feedback, suggestions
    
    def _feedback_for_modality(self, answer: str, tone: str) -> str:
        """生成关于模态的具体反馈"""
        if tone == "赞赏":
            return "你很好地识别了论文中使用的多种模态，并清晰解释了它们的融合方式。分析非常到位。"
        elif tone == "积极":
            return "你正确识别了主要模态类型，解释了基本的融合方法。可以更详细地分析各模态之间的交互机制。"
        elif tone == "中性":
            return "你提到了一些模态类型，但融合方法的描述不够具体。建议仔细阅读论文中关于模态融合的部分。"
        else:
            return "你的回答缺乏对论文中模态类型的准确识别，融合方法也没有具体说明。请重新阅读论文相关章节。"
    
    def _feedback_for_database(self, answer: str, tone: str) -> str:
        """生成关于数据库的具体反馈"""
        if tone == "赞赏":
            return "你详细分析了数据库构建的技术细节，包括索引方法和特征提取过程，非常全面。"
        elif tone == "积极":
            return "你描述了主要的数据库构建方法，提到了一些关键技术。可以更深入分析索引结构的效率和创新点。"
        elif tone == "中性":
            return "你提到了基本的数据库构建方法，但缺少具体的技术细节。建议关注论文中关于索引结构的具体描述。"
        else:
            return "你的回答对数据库构建方式的描述过于笼统，缺乏关键细节。请重点关注论文中的技术实现部分。"
    
    def _feedback_for_experiments(self, answer: str, tone: str) -> str:
        """生成关于实验的具体反馈"""
        if tone == "赞赏":
            return "你全面分析了实验设置、数据集和评估指标，并清晰呈现了结果比较，论述有力。"
        elif tone == "积极":
            return "你提到了主要的实验设置和结果，内容基本准确。可以更详细地分析不同条件下的性能差异和消融实验。"
        elif tone == "中性":
            return "你概述了实验部分，但缺乏对结果的深入分析。建议关注论文中的对比实验和性能指标详情。"
        else:
            return "你的回答对实验部分的描述不够具体，缺少关键的数据集和指标信息。请重新查看论文的实验章节。"
    
    def _feedback_for_innovation(self, answer: str, tone: str) -> str:
        """生成关于创新点的具体反馈"""
        if tone == "赞赏":
            return "你准确把握了论文的核心创新点，并清晰解释了它们如何解决现有问题，分析深入到位。"
        elif tone == "积极":
            return "你列举了主要的创新点，解释较为清晰。可以更深入分析这些创新相对于现有方法的优势和意义。"
        elif tone == "中性":
            return "你提到了一些创新点，但描述较为笼统。建议更具体地说明这些创新如何解决特定问题。"
        else:
            return "你的回答未能准确识别论文的关键创新点。请重新阅读论文，特别关注作者强调的贡献部分。"
    
    def _feedback_general(self, answer: str, tone: str) -> str:
        """生成通用反馈"""
        if tone == "赞赏":
            return "你的回答全面、准确，展示了对论文内容的深入理解，条理清晰，重点突出。"
        elif tone == "积极":
            return "你的回答涵盖了主要内容，表述基本准确。可以进一步提高回答的结构性和逻辑性。"
        elif tone == "中性":
            return "你的回答包含了一些相关信息，但组织不够条理，有些关键点没有充分展开。"
        else:
            return "你的回答缺乏重点，对论文内容的把握不够准确。建议重新阅读论文，梳理主要内容。"


class DiscussionManager:
    """
    讨论管理器，管理学生和教师代理之间的讨论
    """
    def __init__(self, output_dir: str = "discussions"):
        self.reading_agent = ReadingAgent()
        self.student_agent = StudentAgent(self.reading_agent)
        self.teacher_agent = TeacherAgent()
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"DiscussionManager初始化完成，输出目录：{output_dir}")
    
    def conduct_discussion(self, paper_id: str, topics: Optional[List[str]] = None, 
                           rounds_per_topic: int = 2) -> Dict[str, Any]:
        """
        进行一次完整的论文讨论
        
        参数:
            paper_id: 论文ID
            topics: 讨论主题列表，如果为None则使用默认主题
            rounds_per_topic: 每个主题的讨论轮数
            
        返回:
            讨论结果
        """
        # 设置讨论主题
        if topics is None:
            # 随机选择3-5个问题
            num_topics = random.randint(3, 5)
            topics = random.sample(DISCUSSION_TOPICS, min(num_topics, len(DISCUSSION_TOPICS)))
        
        # 读取论文
        paper = self.reading_agent.read_paper(paper_id)
        if paper.get("status") != "success":
            logger.error(f"无法读取论文：{paper_id}")
            return {
                "status": "error",
                "message": f"无法读取论文：{paper_id}",
                "paper_id": paper_id
            }
        
        paper_obj = paper.get("paper", {})
        paper_title = paper_obj.get("title", "Unknown Title")
        
        logger.info(f"开始讨论论文：{paper_title}")
        
        # 初始化讨论记录
        discussion_record = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "discussions": []
        }
        
        # 对每个主题进行讨论
        for topic_idx, topic in enumerate(topics):
            logger.info(f"讨论主题 {topic_idx+1}/{len(topics)}: {topic}")
            
            topic_discussion = []
            prev_feedback = None
            
            # 多轮讨论
            for round_num in range(1, rounds_per_topic + 1):
                logger.info(f"第 {round_num} 轮讨论")
                
                # 学生回答
                student_response = self.student_agent.answer_question(
                    paper, topic, prev_feedback
                )
                
                # 教师评价
                teacher_evaluation = self.teacher_agent.evaluate_answer(
                    paper, topic, student_response, round_num
                )
                
                # 记录本轮讨论
                round_record = {
                    "round": round_num,
                    "topic": topic,
                    "student": student_response,
                    "teacher": teacher_evaluation
                }
                
                topic_discussion.append(round_record)
                
                # 更新前一轮反馈，用于下一轮讨论
                prev_feedback = teacher_evaluation
            
            # 将本主题讨论添加到总记录
            discussion_record["discussions"].extend(topic_discussion)
        
        # 保存讨论记录
        self._save_discussion(discussion_record)
        
        return {
            "status": "success",
            "message": f"成功完成论文'{paper_title}'的讨论",
            "paper_id": paper_id,
            "paper_title": paper_title,
            "topics_count": len(topics),
            "rounds_per_topic": rounds_per_topic,
            "total_rounds": len(discussion_record["discussions"])
        }
    
    def _save_discussion(self, discussion: Dict[str, Any]) -> None:
        """保存讨论记录到JSON文件"""
        paper_id = discussion.get("paper_id", "unknown")
        filename = os.path.join(self.output_dir, f"discussion_{paper_id}.json")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(discussion, f, ensure_ascii=False, indent=2)
            logger.info(f"讨论记录已保存到：{filename}")
        except Exception as e:
            logger.error(f"保存讨论记录失败：{str(e)}")


# 测试代码
if __name__ == "__main__":
    # 创建讨论管理器
    manager = DiscussionManager()
    
    # 测试论文ID (可以替换为实际论文ID)
    paper_id = "2304.01373"
    
    # 进行讨论
    result = manager.conduct_discussion(paper_id)
    
    # 输出结果
    if result["status"] == "success":
        print(f"成功完成论文讨论！")
        print(f"论文标题: {result['paper_title']}")
        print(f"讨论主题数: {result['topics_count']}")
        print(f"每个主题轮数: {result['rounds_per_topic']}")
        print(f"总讨论轮数: {result['total_rounds']}")
        print(f"讨论记录保存在: {os.path.join(manager.output_dir, f'discussion_{paper_id}.json')}")
    else:
        print(f"讨论失败: {result['message']}") 