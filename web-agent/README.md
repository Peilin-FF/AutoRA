# 多模态RAG文献研究助手

## 项目概述

多模态RAG文献研究助手是一个专为研究人员和学者设计的工具，旨在简化多模态检索增强生成(Multimodal Retrieval-Augmented Generation)领域的文献收集、阅读和分析工作。

### 主要功能

1. **文献搜索**：从arXiv等学术网站搜索最新的多模态RAG相关论文
2. **全文检索**：获取并解析论文PDF，提取全文内容
3. **元数据提取**：自动提取论文的标题、作者、发布日期、引用次数等信息
4. **多模态分析**：识别论文中涉及的模态类型（文本、图像、视频、音频等）
5. **技术要点提取**：分析数据库构建方法、实验设计与结果等核心内容
6. **学生-教师讨论模式**：通过模拟学生提问和教师点评的方式深入理解论文内容

## 安装与配置

### 前提条件

- Python 3.8 或更高版本
- pip (Python包管理器)

### 安装步骤

1. 克隆代码库或下载源代码：
   ```
   git clone <repository_url>
   cd AutoRA/web-agent
   ```

2. 使用自动化脚本安装和运行演示：

   **Windows**:
   ```
   demo.bat
   ```

   **Linux/macOS**:
   ```
   chmod +x demo.sh
   ./demo.sh
   ```

   或者手动安装依赖：
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

## 使用方法

### 命令行界面

该工具提供了简单易用的命令行界面，支持以下操作：

#### 基本搜索与分析

```
python run_demo.py --query "multimodal RAG" --max-results 20 --output results.json
```

参数说明：
- `--query`: 搜索查询字符串
- `--max-results`: 最大结果数量
- `--output`: 输出文件路径

#### 启用论文讨论功能

```
python run_demo.py --with-discussion --discussion-rounds 2 --discussion-dir discussions
```

参数说明：
- `--with-discussion`: 启用学生-教师讨论功能
- `--discussion-rounds`: 每个讨论主题的轮数
- `--discussion-dir`: 讨论记录保存目录

### 在Python代码中使用

也可以在自己的Python代码中使用本工具提供的功能：

```python
from multimodal_rag_demo import MultimodalRAGAnalyzer
from discussion_agents import DiscussionManager

# 创建分析器
analyzer = MultimodalRAGAnalyzer()

# 运行分析
results = analyzer.run_analysis("multimodal RAG retrieval", max_results=10)

# 选择一篇论文进行讨论
paper_id = results["papers"][0]["paper_id"]
discussion_manager = DiscussionManager()
discussion = discussion_manager.conduct_discussion(paper_id, rounds_per_topic=2)
```

## 论文讨论功能详解

多模态RAG文献研究助手引入了模拟学术讨论的功能，通过学生代理和教师代理的交互，帮助深入理解论文内容。

### 讨论流程

1. **学生代理** 首先阅读论文并尝试回答关于论文的问题，如"论文中使用了哪些模态"、"论文的创新点是什么"等
2. **教师代理** 评估学生的回答，提供评分和建设性反馈
3. 根据教师的反馈，**学生代理** 改进自己的回答
4. 整个讨论过程被记录下来，保存为JSON格式

### 讨论内容示例

```json
{
  "paper_id": "2304.01373",
  "paper_title": "MultiModal-GPT: A Vision and Language Model for Multimodal Generation",
  "discussions": [
    {
      "round": 1,
      "topic": "这篇论文提出的多模态RAG方法有哪些创新点？",
      "student": {
        "answer": "这篇论文的主要创新点包括：新的多模态融合机制; 改进的跨模态检索算法。论文详细分析了这些创新如何解决现有多模态RAG系统中的挑战，特别是在处理模态间信息不平衡和跨模态语义对齐方面取得了进展。",
        "reasoning": "我首先分析了论文的摘要和相关章节，提取了与问题'这篇论文提出的多模态RAG方法有哪些创新点？'相关的关键信息。",
        "confidence": 0.72
      },
      "teacher": {
        "feedback": "良好。你列举了主要的创新点，解释较为清晰。可以更深入分析这些创新相对于现有方法的优势和意义。",
        "suggestions": "可以添加更多关于论文方法的技术细节，以及与现有工作的比较",
        "score": 82.5
      }
    },
    {
      "round": 2,
      "topic": "这篇论文提出的多模态RAG方法有哪些创新点？",
      "student": {
        "answer": "这篇论文的主要创新点包括：新的多模态融合机制，采用了注意力机制实现文本和视觉特征的动态融合; 改进的跨模态检索算法，引入了语义一致性约束; 高效的多模态索引结构，支持大规模数据检索。与现有方法相比，该方法在MSCOCO等数据集上取得了10-15%的性能提升，尤其在处理模态间信息不平衡方面表现突出。论文详细分析了这些创新如何解决现有多模态RAG系统中的挑战，通过消融实验验证了各组件的有效性。",
        "reasoning": "我首先分析了论文的摘要和相关章节，提取了与问题'这篇论文提出的多模态RAG方法有哪些创新点？'相关的关键信息。 根据老师之前的反馈'良好。你列举了主要的创新点，解释较为清晰。可以更深入分析这些创新相对于现有方法的优势和意义。'，我特别注意了可以添加更多关于论文方法的技术细节，以及与现有工作的比较。",
        "confidence": 0.84
      },
      "teacher": {
        "feedback": "非常好。你准确把握了论文的核心创新点，并清晰解释了它们如何解决现有问题，分析深入到位。",
        "suggestions": "你的回答已经很好，可以尝试从更广阔的背景角度进行讨论",
        "score": 92.7
      }
    }
  ]
}
```

## 项目结构

- `agents.py`: 定义ArxivAgent等智能代理
- `tools.py`: 实现论文搜索和检索功能
- `reading_agents.py`: 实现论文阅读理解功能
- `discussion_agents.py`: 实现学生-教师讨论功能
- `run_demo.py`: 演示脚本
- `multimodal_rag_demo.py`: 多模态RAG分析器
- `demo.bat`/`demo.sh`: 自动安装和运行脚本
- `requirements.txt`: 项目依赖

## 注意事项

- 本工具遵循arXiv的API使用政策，请勿过度频繁地请求以避免被封禁
- 目前主要支持英文论文，对其他语言的支持有限
- 论文分析和讨论功能基于规则和模板，不依赖外部LLM API，因此分析深度有一定局限

## 未来计划

- 支持更多学术网站和数据库
- 增强多模态理解能力
- 添加文献综述自动生成功能
- 实现基于外部LLM API的更深入分析

## 贡献与反馈

欢迎通过Issue或Pull Request提供反馈和贡献代码。

## 许可证

[MIT License](LICENSE) 