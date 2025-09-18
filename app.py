import streamlit as st
import pandas as pd
from main import (
    analyze_webpage_links,
    adaptive_analysis_with_intelligent_params,
    intelligent_parameter_estimation,
    get_client
)
import os
import tempfile
import concurrent.futures
import time
import requests
import re
from bs4 import BeautifulSoup
import logging
import json

# 设置日志
logger = logging.getLogger(__name__)

st.set_page_config(page_title="教授研究兴趣分析器 🎓", layout="wide")

# 初始化会话状态变量
if "professor_results" not in st.session_state:
    st.session_state.professor_results = None
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "similarity_results" not in st.session_state:
    st.session_state.similarity_results = None


def render_keyword_tag(keyword: str, highlight: bool = False) -> str:
    """渲染关键词标签，支持高亮显示"""
    base_style = "display: inline-block; margin: 2px; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;"
    
    if highlight:
        # 匹配的关键词用高亮颜色
        style = base_style + "background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"
    else:
        # 普通关键词用灰色
        style = base_style + "background: #f0f2f6; color: #262730; border: 1px solid #e6eaed;"
    
    return f"<span style='{style}'>{keyword}</span>"


def calculate_similarity(prof_interests, user_interests, api_key):
    """使用LLM计算教授研究兴趣与用户兴趣的相似度"""
    try:
        # 创建客户端
        current_client = get_client(api_key)

        llm_response = current_client.chat.completions.create(
            model="doubao-seed-1-6-250615",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的学术匹配专家。你的任务是分析教授的研究兴趣与用户的研究兴趣之间的相似度，给出0-100的分数和详细解释。",
                },
                {
                    "role": "user",
                    "content": f"请分析以下教授的研究兴趣与用户的研究兴趣之间的相似度。给出0-100的相似度分数和详细解释。\n\n教授研究兴趣: {prof_interests}\n\n用户研究兴趣: {user_interests}",
                },
            ],
        )
        return llm_response.choices[0].message.content.strip()
    except Exception as e:
        return f"分析相似度时出错: {str(e)}"


def extract_structured_similarity_data(similarity_text):
    """从相似度分析文本中提取结构化数据"""
    try:
        # 清理响应文本，移除可能的markdown标记
        cleaned_text = similarity_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        # 尝试解析JSON
        try:
            result = json.loads(cleaned_text)
            
            # 验证JSON结构
            if validate_similarity_structure(result):
                return {
                    'success': True,
                    'data': result,
                    'overall_score': result.get('overall_similarity', 0),
                    'dimension_scores': result.get('dimension_scores', {}),
                    'matched_keywords': result.get('matched_keywords', []),
                    'reasoning': result.get('reasoning', {}),
                    'confidence': result.get('confidence', 0.5)
                }
        except json.JSONDecodeError:
            pass
        
        # JSON解析失败，使用fallback机制
        fallback_result = extract_fallback_similarity_data(similarity_text)
        return {
            'success': False,
            'data': fallback_result,
            'overall_score': fallback_result.get('overall_similarity', 0),
            'dimension_scores': {},
            'matched_keywords': [],
            'reasoning': {'overall': similarity_text},
            'confidence': 0.3
        }
        
    except Exception as e:
        # 出错时返回默认值
        return {
            'success': False,
            'data': {},
            'overall_score': 0,
            'dimension_scores': {},
            'matched_keywords': [],
            'reasoning': {'overall': f'解析错误: {str(e)}'},
            'confidence': 0.0
        }


def validate_similarity_structure(data):
    """验证相似度分析结果的JSON结构是否正确"""
    if not isinstance(data, dict):
        return False
    
    required_keys = ['overall_similarity', 'dimension_scores', 'reasoning']
    if not all(key in data for key in required_keys):
        return False
    
    # 验证分数范围
    overall = data.get('overall_similarity')
    if not isinstance(overall, (int, float)) or not (0 <= overall <= 100):
        return False
    
    # 验证维度分数
    dimension_scores = data.get('dimension_scores', {})
    if not isinstance(dimension_scores, dict):
        return False
    
    expected_dimensions = ['research_topics', 'research_methods', 'theoretical_framework', 
                          'application_domains', 'keyword_matching']
    
    for dim in expected_dimensions:
        score = dimension_scores.get(dim)
        if score is not None and (not isinstance(score, (int, float)) or not (0 <= score <= 100)):
            return False
    
    # 验证置信度
    confidence = data.get('confidence')
    if confidence is not None and (not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1)):
        return False
    
    return True


def extract_fallback_similarity_data(similarity_text):
    """当JSON解析失败时的备用数据提取机制"""
    import re
    
    # 尝试查找分数模式
    score_patterns = [
        r"相似度[：:]\s*(\d+)",
        r"总体相似度[：:]\s*(\d+)",
        r"overall_similarity[\"'\s]*[：:]\s*(\d+)",
        r"(\d+)\s*分",
        r"(\d+)/100"
    ]
    
    extracted_score = 0
    for pattern in score_patterns:
        matches = re.findall(pattern, similarity_text, re.IGNORECASE)
        if matches:
            try:
                score = int(matches[0])
                if 0 <= score <= 100:
                    extracted_score = score
                    break
            except ValueError:
                continue
    
    # 如果没有找到有效分数，尝试提取所有数字并选择最合理的
    if extracted_score == 0:
        all_numbers = re.findall(r'\b(\d{1,3})\b', similarity_text)
        valid_scores = [int(num) for num in all_numbers if 0 <= int(num) <= 100]
        if valid_scores:
            # 选择中位数作为最可能的分数
            valid_scores.sort()
            extracted_score = valid_scores[len(valid_scores) // 2]
    
    return {
        'overall_similarity': extracted_score,
        'dimension_scores': {},
        'matched_keywords': [],
        'reasoning': {'overall': similarity_text},
        'confidence': 0.3
    }


def calculate_advanced_similarity(prof_interests, prof_keywords, user_interests, api_key):
    """使用LLM进行多维度研究兴趣相似度分析"""
    try:
        # 创建客户端
        current_client = get_client(api_key)

        # 处理教授关键词
        keywords_str = ", ".join(prof_keywords) if prof_keywords and isinstance(prof_keywords, list) else "无关键词"

        # 构建结构化提示词
        system_prompt = """你是一个专业的学术匹配分析专家。你需要从多个维度分析教授的研究兴趣与用户研究兴趣的匹配度。

请按照以下5个维度进行评分（每个维度0-100分）：

1. **研究主题相似度**: 研究的核心话题、问题领域的重叠程度
2. **研究方法匹配度**: 研究方法论、技术手段、分析工具的相似性  
3. **理论框架重叠度**: 理论基础、学科背景、概念框架的契合度
4. **应用领域契合度**: 实际应用场景、目标群体、解决问题的相似性
5. **关键词精确匹配**: 具体术语、专业词汇的直接匹配程度

请严格按照以下JSON格式返回结果：
{
    "overall_similarity": 整体相似度分数(0-100),
    "dimension_scores": {
        "research_topics": 研究主题分数(0-100),
        "research_methods": 研究方法分数(0-100), 
        "theoretical_framework": 理论框架分数(0-100),
        "application_domains": 应用领域分数(0-100),
        "keyword_matching": 关键词匹配分数(0-100)
    },
    "matched_keywords": ["匹配的关键词1", "匹配的关键词2"],
    "reasoning": {
        "strengths": "匹配优势的具体说明",
        "gaps": "存在差距的具体分析", 
        "overall": "综合评价和建议"
    },
    "confidence": 置信度(0.0-1.0)
}

注意：
- 所有分数必须是0-100之间的整数
- 整体相似度应该是各维度分数的加权平均
- 置信度反映分析的可靠程度
- 匹配关键词应从教授关键词中选择与用户兴趣相关的词汇"""

        user_prompt = f"""请分析以下教授与用户的研究兴趣匹配度：

**教授研究兴趣:**
{prof_interests}

**教授关键词:**  
{keywords_str}

**用户研究兴趣:**
{user_interests}

请按照要求的JSON格式进行多维度分析。"""

        llm_response = current_client.chat.completions.create(
            model="doubao-seed-1-6-250615",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1  # 降低温度以获得更一致的结果
        )
        
        return llm_response.choices[0].message.content.strip()
    except Exception as e:
        return f"多维度相似度分析时出错: {str(e)}"


# 添加一个函数来显示结果，避免代码重复
def display_results(results_df):
    """显示相似度结果"""
    if results_df is None or len(results_df) == 0:
        st.warning("⚠️ 没有结果可显示")
        return

    for _, row in results_df.iterrows():
        with st.expander(f"🔗 相似度 {row['Score']}分 - {row.get('Professor Name', 'Unknown')}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 👨‍🏫 教授信息")
                st.write(f"**姓名:** {row.get('Professor Name', 'N/A')}")
                st.write(f"**职位:** {row.get('Title', 'N/A')}")
                st.write(f"**院系:** {row.get('Department', 'N/A')}")
                st.write(f"**网址:** {row['URL']}")
                
                st.markdown("#### 📖 研究兴趣")
                st.write(row["Research Interests"])
                
                st.markdown("#### 📊 相似度分析")
                st.write(row["Similarity Analysis"])
            
            with col2:
                st.markdown("#### 🏷️ 关键词")
                keywords = row.get('Keywords', [])
                if keywords and isinstance(keywords, list):
                    for keyword in keywords:
                        st.markdown(render_keyword_tag(keyword), unsafe_allow_html=True)
                else:
                    st.write("无关键词")
                
                st.markdown("#### 📎 相关链接")
                additional_urls = row.get('Additional URLs', [])
                if additional_urls and isinstance(additional_urls, list):
                    for url in additional_urls[:3]:  # 限制显示3个相关链接
                        st.markdown(f"[相关页面]({url})")
                else:
                    st.write("无相关链接")
                
                if row.get('Confidence Score'):
                    st.markdown("#### 🎯 置信度")
                    confidence = float(row.get('Confidence Score', 0))
                    st.progress(confidence)
                    st.write(f"{confidence:.1%}")


def display_professor_results(results: list, key_prefix: str = "default"):
    """显示教授分析结果"""
    if not results:
        st.warning("⚠️ 没有找到教授页面")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 现在results中只包含教授页面，不需要过滤
    professor_results = results
    
    st.subheader(f"👨‍🏫 找到的教授页面 ({len(professor_results)}位)")
    
    # 添加过滤和排序选项 - 使用动态key
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence_filter = st.slider("最低置信度", 0.0, 1.0, 0.0, 0.1, key=f"confidence_filter_{key_prefix}")
    with col2:
        sort_by = st.selectbox("排序方式", ["置信度", "教授姓名", "院系"], key=f"sort_by_{key_prefix}")
    with col3:
        sort_order = st.selectbox("排序顺序", ["降序", "升序"], key=f"sort_order_{key_prefix}")
    
    # 应用过滤
    filtered_results = [r for r in professor_results if r.get('Confidence Score', 0) >= confidence_filter]
    
    # 应用排序
    if sort_by == "置信度":
        sort_key = lambda x: x.get('Confidence Score', 0)
    elif sort_by == "教授姓名":
        sort_key = lambda x: x.get('Professor Name', '')
    else:
        sort_key = lambda x: x.get('Department', '')
    
    ascending = (sort_order == "升序")
    filtered_results = sorted(filtered_results, key=sort_key, reverse=not ascending)
    
    # 显示结果
    for result in filtered_results:
        # 组装标题徽标
        title_badges = []
        if result.get('PhD Not Recruiting', False):
            title_badges.append("❌ 不招博士生")
        if result.get('Insufficient Content', False):
            title_badges.append("⚠️ 内容不足")
        badges_str = ("  ".join(title_badges)) if title_badges else ""

        display_title = f"👨‍🏫 {result.get('Professor Name', 'Unknown')} - {result.get('Title', '')}"
        if badges_str:
            display_title = f"{display_title}  |  {badges_str}"

        with st.expander(display_title):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 📍 基本信息")
                st.write(f"**姓名:** {result.get('Professor Name', 'N/A')}")
                st.write(f"**职位:** {result.get('Title', 'N/A')}")
                st.write(f"**院系:** {result.get('Department', 'N/A')}")
                st.write(f"**网址:** {result.get('URL', 'N/A')}")
                
                st.markdown("#### 📖 研究兴趣")
                research_interests = result.get('Research Interests', '')
                if research_interests:
                    st.write(research_interests)
                else:
                    st.write("未找到研究兴趣信息")

                # 高亮状态区
                if result.get('PhD Not Recruiting', False):
                    st.markdown("#### ❌ 不招博士生")
                    evidence = result.get('PhD Evidence', '')
                    if evidence:
                        st.info(evidence)
                    else:
                        st.info("页面明确表示当前不招收博士生")
                if result.get('Insufficient Content', False):
                    st.markdown("#### ⚠️ 内容不足")
                    reasons = result.get('Insufficient Reasons', []) or []
                    if reasons:
                        # 将枚举转换为可读标签
                        reason_map = {
                            'too_short_text': '页面文本过短',
                            'no_research_section': '缺少研究相关板块',
                            'few_paragraphs_no_keywords': '段落过少且缺少研究关键词',
                            'mostly_contact_admin': '主要为联系/行政信息',
                            'exception_during_extraction': '提取过程中发生异常'
                        }
                        readable = [reason_map.get(r, r) for r in reasons]
                        st.warning("；".join(readable))
                    else:
                        st.warning("页面可用信息不足")
            
            with col2:
                st.markdown("#### 🏷️ 关键词")
                keywords = result.get('Keywords', [])
                if keywords and isinstance(keywords, list) and len(keywords) > 0:
                    for keyword in keywords:
                        st.markdown(render_keyword_tag(keyword), unsafe_allow_html=True)
                else:
                    st.write("无关键词")
                
                st.markdown("#### 📎 相关链接")
                additional_urls = result.get('Additional URLs', [])
                if additional_urls and isinstance(additional_urls, list) and len(additional_urls) > 0:
                    for url in additional_urls[:3]:  # 限制显示3个相关链接
                        st.markdown(f"[相关页面]({url})")
                else:
                    st.write("无相关链接")
                
                st.markdown("#### 🎯 置信度")
                confidence = result.get('Confidence Score', 0)
                st.progress(confidence)
                st.write(f"{confidence:.1%}")
    
    # 提供下载功能
    st.markdown("---")
    df_download = pd.DataFrame(professor_results)
    csv_data = df_download.to_csv(index=False)
    st.download_button(
        label="📥 下载教授信息",
        data=csv_data,
        file_name="professor_analysis_results.csv",
        mime="text/csv",
        key=f"download_button_{key_prefix}"
    )

    # 显示分析统计
    st.markdown("---")
    st.markdown("#### 📊 分析统计")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("找到教授", len(professor_results))
    with col2:
        avg_confidence = sum(r.get('Confidence Score', 0) for r in professor_results) / len(professor_results) if professor_results else 0
        st.metric("平均置信度", f"{avg_confidence:.1%}")
    with col3:
        with_keywords = sum(1 for r in professor_results if r.get('Keywords') and len(r.get('Keywords', [])) > 0)
        st.metric("含关键词", f"{with_keywords}/{len(professor_results)}")


def display_advanced_similarity_results(results_df):
    """显示多维度相似度分析结果"""
    if results_df is None or len(results_df) == 0:
        st.warning("⚠️ 没有结果可显示")
        return

    # 按分数排序
    sorted_df = results_df.sort_values(by="Score", ascending=False)

    for _, row in sorted_df.iterrows():
        # 获取结构化数据
        similarity_text = row.get("Similarity Analysis", "")
        similarity_data = extract_structured_similarity_data(similarity_text)
        
        # 标题显示总体相似度和置信度
        confidence_indicator = "🔥" if similarity_data['confidence'] > 0.8 else "✅" if similarity_data['confidence'] > 0.5 else "⚠️"
        title = f"{confidence_indicator} 相似度 {row['Score']}分 - {row.get('Professor Name', 'Unknown')}"
        
        with st.expander(title):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### 👨‍🏫 教授信息")
                st.write(f"**姓名:** {row.get('Professor Name', 'N/A')}")
                st.write(f"**职位:** {row.get('Title', 'N/A')}")
                st.write(f"**院系:** {row.get('Department', 'N/A')}")
                st.write(f"**网址:** {row['URL']}")
                
                st.markdown("#### 📖 研究兴趣")
                research_text = row["Research Interests"]
                matched_keywords = similarity_data.get('matched_keywords', [])
                if matched_keywords:
                    highlighted_research = highlight_matched_keywords(research_text, matched_keywords)
                    st.markdown(highlighted_research, unsafe_allow_html=True)
                else:
                    st.write(research_text)
                
                # 显示多维度分析
                if similarity_data['success'] and similarity_data['dimension_scores']:
                    st.markdown("#### 📊 多维度匹配分析")
                    
                    # 尝试显示雷达图
                    radar_fig = render_dimension_radar_chart(similarity_data['dimension_scores'])
                    if radar_fig:
                        # 使用教授URL作为唯一key
                        chart_key = f"radar_chart_{hash(row['URL'])}"
                        st.plotly_chart(radar_fig, use_container_width=True, key=chart_key)
                    else:
                        # 如果没有plotly，使用文本显示
                        render_dimension_scores(similarity_data['dimension_scores'])
                    
                    # 显示详细推理
                    reasoning = similarity_data['reasoning']
                    if isinstance(reasoning, dict):
                        if reasoning.get('strengths'):
                            st.markdown("**🎯 匹配优势:**")
                            st.write(reasoning['strengths'])
                        if reasoning.get('gaps'):
                            st.markdown("**🔍 待改进点:**")
                            st.write(reasoning['gaps'])
                        if reasoning.get('overall'):
                            st.markdown("**💡 综合评价:**")
                            st.write(reasoning['overall'])
                    else:
                        st.markdown("#### 📝 详细分析")
                        st.write(similarity_data['reasoning'].get('overall', similarity_text))
                else:
                    st.markdown("#### 📝 分析结果")
                    st.write(similarity_text)
            
            with col2:
                # 显示匹配关键词
                st.markdown("#### 🔗 匹配关键词")
                matched_keywords = similarity_data.get('matched_keywords', [])
                if matched_keywords and isinstance(matched_keywords, list):
                    for keyword in matched_keywords:
                        st.markdown(render_keyword_tag(keyword, highlight=True), unsafe_allow_html=True)
                else:
                    st.write("无匹配关键词")
                
                # 显示教授所有关键词
                st.markdown("#### 🏷️ 教授关键词")
                prof_keywords = row.get('Keywords', [])
                if prof_keywords and isinstance(prof_keywords, list):
                    for keyword in prof_keywords:
                        is_matched = keyword in matched_keywords if matched_keywords else False
                        st.markdown(render_keyword_tag(keyword, highlight=is_matched), unsafe_allow_html=True)
                else:
                    st.write("无关键词")
                
                # 显示置信度
                st.markdown("#### 🎯 分析置信度")
                confidence = similarity_data.get('confidence', 0)
                st.progress(confidence)
                st.write(f"{confidence:.1%}")
                
                # 显示相关链接
                st.markdown("#### 📎 相关链接")
                additional_urls = row.get('Additional URLs', [])
                if additional_urls and isinstance(additional_urls, list):
                    for i, url in enumerate(additional_urls[:3]):
                        st.markdown(f"[相关页面 {i+1}]({url})")
                else:
                    st.write("无相关链接")


def render_dimension_scores(dimension_scores):
    """渲染维度分数显示"""
    dimension_names = {
        'research_topics': '🎯 研究主题',
        'research_methods': '🔬 研究方法', 
        'theoretical_framework': '📚 理论框架',
        'application_domains': '🌍 应用领域',
        'keyword_matching': '🔗 关键词匹配'
    }
    
    # 创建两列布局显示维度分数
    col1, col2 = st.columns(2)
    
    dimensions = list(dimension_scores.keys())
    for i, (dim_key, score) in enumerate(dimension_scores.items()):
        display_name = dimension_names.get(dim_key, dim_key)
        
        # 交替显示在两列中
        with col1 if i % 2 == 0 else col2:
            # 使用进度条和分数显示
            st.metric(label=display_name, value=f"{score}分")
            st.progress(score / 100.0)


def render_dimension_radar_chart(dimension_scores):
    """渲染五维度雷达图"""
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        # 维度名称映射
        dimension_names = {
            'research_topics': '研究主题',
            'research_methods': '研究方法', 
            'theoretical_framework': '理论框架',
            'application_domains': '应用领域',
            'keyword_matching': '关键词匹配'
        }
        
        # 提取分数和标签
        labels = []
        values = []
        for key, score in dimension_scores.items():
            labels.append(dimension_names.get(key, key))
            values.append(score)
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(59, 130, 246, 0.8)', width=2),
            marker=dict(color='rgba(59, 130, 246, 1)', size=8),
            name='匹配度'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10),
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except ImportError:
        # 如果没有plotly，返回None
        return None


def highlight_matched_keywords(text, keywords):
    """在文本中高亮匹配的关键词"""
    if not keywords or not text:
        return text
    
    import re
    highlighted_text = text
    
    # 对每个关键词进行高亮处理
    for keyword in keywords:
        if keyword and len(keyword.strip()) > 0:
            # 使用正则表达式进行大小写不敏感的匹配
            pattern = re.compile(re.escape(keyword.strip()), re.IGNORECASE)
            highlighted_text = pattern.sub(
                f'<mark style="background-color: #FFE066; padding: 1px 3px; border-radius: 3px;">{keyword}</mark>',
                highlighted_text
            )
    
    return highlighted_text


def validate_similarity_scores(similarity_data):
    """验证相似度分数的一致性"""
    if not similarity_data.get('success', False):
        return True  # 如果解析失败，跳过验证
    
    overall_score = similarity_data.get('overall_score', 0)
    dimension_scores = similarity_data.get('dimension_scores', {})
    
    # 基本范围检查
    if not (0 <= overall_score <= 100):
        return False
    
    # 检查各维度分数
    for dim, score in dimension_scores.items():
        if not isinstance(score, (int, float)) or not (0 <= score <= 100):
            return False
    
    # 一致性检查：总体分数应该与维度分数相关
    if dimension_scores:
        avg_dimension_score = sum(dimension_scores.values()) / len(dimension_scores)
        # 允许±20分的差异，因为总体分数可能有权重
        if abs(overall_score - avg_dimension_score) > 20:
            return False
    
    return True


def main():
    st.title("🎓 教授研究兴趣分析器")

    # API设置在侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        api_key = st.text_input("输入API密钥（必填）", type="password")
        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API密钥已设置")
        else:
            st.warning("⚠️ 请输入API密钥才能使用分析功能")

    tab1, tab2 = st.tabs(["🔍 网站分析", "🔄 兴趣匹配"])

    with tab1:
        st.header("📊 分析大学网站上的教授页面")

        # 分析模式选择
        st.markdown("#### 🎯 分析模式")
        analysis_mode = st.radio(
            "选择分析模式:",
            [
                "🧠 智能自适应分析 (推荐)",
                "⚙️ 手动参数设置",
                "⚡ 快速分析 (默认参数)"
            ],
            help="智能模式会自动分析页面特征，推荐最优参数，并根据结果动态调整搜索策略"
        )

        if analysis_mode == "🧠 智能自适应分析 (推荐)":
            st.info("💡 **智能模式**: 程序将自动分析页面特征，推荐最优参数，并根据结果动态调整搜索策略")
            use_intelligent_params = True
            show_manual_params = False
            
        elif analysis_mode == "⚙️ 手动参数设置":
            st.info("🔧 **手动模式**: 您可以自定义所有分析参数")
            use_intelligent_params = False
            show_manual_params = True
            
        else:  # 快速分析
            st.info("⚡ **快速模式**: 使用预设的默认参数进行分析")
            use_intelligent_params = False
            show_manual_params = False

        # 只在需要时显示参数推荐功能
        if analysis_mode == "⚙️ 手动参数设置":
            col1, col2 = st.columns([2, 1])

            with col1:
                start_url = st.text_input(
                    "输入起始URL", "https://journalism.uiowa.edu/people", key="start_url_manual"
                )

            with col2:
                st.markdown("#### 🧠 智能参数推荐")
                if st.button("🔍 分析页面并推荐参数", key="analyze_params_button"):
                    if start_url:
                        with st.spinner("正在分析页面特征..."):
                            try:
                                recommendations = intelligent_parameter_estimation(start_url)
                                
                                st.success("✅ 智能分析完成!")
                                st.info(f"**推荐原因**: {recommendations['reasoning']}")
                                
                                # 保存推荐参数到session state
                                st.session_state.recommended_max_links = recommendations['max_links']
                                st.session_state.recommended_max_pages = recommendations['max_pages']
                                st.session_state.page_analysis = {
                                    'page_type': recommendations.get('page_type', 'unknown'),
                                    'professor_density': recommendations.get('professor_density', 0),
                                    'pagination_detected': recommendations.get('pagination_detected', False)
                                }
                                
                            except Exception as e:
                                st.error(f"智能分析失败: {str(e)}")
        else:
            start_url = st.text_input(
                "输入起始URL", "https://journalism.uiowa.edu/people", key="start_url_default"
            )

        # 参数设置部分 - 只在手动模式下显示
        if show_manual_params:
            st.markdown("#### ⚙️ 分析参数设置")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                # 检查是否有推荐参数
                recommended_links = getattr(st.session_state, 'recommended_max_links', 30)
                max_links = st.number_input(
                    "最大分析链接数", 
                    min_value=10, 
                    max_value=500, 
                    value=recommended_links, 
                    step=10,
                    help="智能推荐的链接数，可手动调整",
                    key="max_links_manual"
                )
                
            with col4:
                recommended_pages = getattr(st.session_state, 'recommended_max_pages', 3)
                max_pages = st.number_input(
                    "最多跟随页数", 
                    min_value=1, 
                    max_value=10, 
                    value=recommended_pages, 
                    step=1,
                    help="智能推荐的页面数，可手动调整",
                    key="max_pages_manual"
                )
                
            with col5:
                max_workers = st.number_input(
                    "工作线程数", min_value=1, max_value=10, value=5, step=1, key="max_workers_manual"
                )
            
            # 显示页面分析结果
            if hasattr(st.session_state, 'page_analysis'):
                analysis = st.session_state.page_analysis
                
                st.markdown("#### 📊 页面分析结果")
                
                col6, col7, col8 = st.columns(3)
                with col6:
                    page_type_display = {
                        'department': '🏛️ 系级页面',
                        'college': '🏫 学院级页面', 
                        'faculty_list': '👥 教授列表',
                        'unknown': '❓ 未知类型'
                    }
                    st.metric("页面类型", page_type_display.get(analysis['page_type'], '未知'))
                    
                with col7:
                    density = analysis.get('professor_density', 0)
                    st.metric("教授链接密度", f"{density:.1%}")
                    
                with col8:
                    pagination_status = "✅ 检测到" if analysis.get('pagination_detected') else "❌ 未检测到"
                    st.metric("分页结构", pagination_status)
        else:
            # 在智能模式下仍然提供基本的线程数设置
            max_workers = st.number_input(
                "工作线程数", min_value=1, max_value=10, value=5, step=1,
                help="并发处理的线程数量", key="max_workers_auto"
            )

        if st.button("🚀 开始分析", key="start_analysis_button"):
            if not st.session_state.api_key:
                st.error("❌ 请在侧边栏设置 OpenAI API 密钥")
            elif not start_url:
                st.error("❌ 请输入起始URL")
            else:
                try:
                    with st.spinner("正在分析教授页面..."):
                        progress_bar = st.progress(0)
                        
                        # 根据分析模式选择不同的函数
                        if analysis_mode == "🧠 智能自适应分析 (推荐)":
                            progress_bar.progress(20)
                            
                            results = adaptive_analysis_with_intelligent_params(
                                start_url, 
                                st.session_state.api_key, 
                                max_workers,
                                use_intelligent_params=True
                            )
                            
                        elif analysis_mode == "⚙️ 手动参数设置":
                            progress_bar.progress(20)
                            
                            # 确保参数已定义
                            if 'max_links' not in locals():
                                max_links = getattr(st.session_state, 'recommended_max_links', 30)
                            if 'max_pages' not in locals():
                                max_pages = getattr(st.session_state, 'recommended_max_pages', 3)
                                
                            results = analyze_webpage_links(
                                start_url, 
                                st.session_state.api_key, 
                                max_links, 
                                max_pages, 
                                max_workers
                            )
                            
                        else:  # 快速分析模式
                            progress_bar.progress(20)
                            
                            results = analyze_webpage_links(
                                start_url, 
                                st.session_state.api_key, 
                                30,  # 默认链接数
                                3,   # 默认页面数
                                max_workers
                            )
                        
                        progress_bar.progress(100)
                        
                        if results:
                            st.success(f"✅ 分析完成！找到 {len(results)} 个链接")
                            st.session_state.results = results
                            
                            # 显示分析模式的执行结果
                            if analysis_mode == "🧠 智能自适应分析 (推荐)":
                                professor_count = len([r for r in results if r.get('Is Professor Page') == 'Yes'])
                                st.info(f"🎯 **智能分析结果**: 发现 {professor_count} 位教授，分析了 {len(results)} 个链接")
                            
                            # 显示结果
                            display_professor_results(results, key_prefix="tab1_main")
                            
                        else:
                            st.warning("⚠️ 未找到任何相关链接")
                            
                except Exception as e:
                    st.error(f"❌ 分析失败: {str(e)}")
                    logger.error(f"Analysis failed: {e}", exc_info=True)
                finally:
                    progress_bar.empty()

        # 显示会话中保存的结果
        if hasattr(st.session_state, 'results') and st.session_state.results:
            st.subheader("📋 上次分析结果")
            display_professor_results(st.session_state.results, key_prefix="tab1_saved")

    with tab2:
        st.header("🔄 研究兴趣相似度匹配")

        # 检查是否有教授结果
        professor_results = None
        if hasattr(st.session_state, 'results') and st.session_state.results:
            # 现在results中只包含教授页面
            professor_results = st.session_state.results
            has_professors = len(professor_results) > 0
        else:
            st.info("ℹ️ 请先在'🔍 网站分析'标签中分析一个网站以获取教授研究兴趣")
            professor_results = []
            has_professors = False

        # 用户输入研究兴趣
        user_interests = st.text_area(
            "✏️ 输入您的研究兴趣（请详细描述您感兴趣的研究领域、方法和主题）", height=150, key="user_interests_input"
        )
        has_interests = bool(user_interests.strip())

        # 始终显示按钮，但根据条件禁用
        col1, col2 = st.columns([3, 1])
        with col1:
            calc_button = st.button(
                "🔍 计算相似度", disabled=not has_professors or not has_interests, key="calc_similarity_button"
            )
            if not has_professors and not has_interests:
                st.caption("请先分析网站并输入研究兴趣")
            elif not has_professors:
                st.caption("请先分析网站以获取教授数据")
            elif not has_interests:
                st.caption("请输入您的研究兴趣")
        with col2:
            # 检查是否有相似度结果数据
            has_similarity_results = (
                hasattr(st.session_state, 'similarity_results') and
                st.session_state.similarity_results is not None and
                len(st.session_state.similarity_results) > 0
            )
            
            sort_button = st.button(
                "📊 排序结果",
                disabled=not has_similarity_results,
                key="sort_results_button"
            )
            if not has_similarity_results:
                st.caption("请先计算相似度")

        # 处理计算相似度按钮
        if calc_button and has_professors and has_interests:
            if not st.session_state.api_key:
                st.error("❌ 请先在侧边栏设置API密钥")
            else:
                st.subheader("📊 相似度分析结果")

                # 显示进度指示器
                progress_container = st.empty()
                progress_bar = st.progress(0)
                progress_text = st.empty()

                # 为每位教授计算相似度
                results = []
                total_professors = len(professor_results)

                for i, professor in enumerate(professor_results):
                    # 更新进度
                    progress = (i) / total_professors
                    progress_bar.progress(progress)
                    progress_text.text(
                        f"⏳ 正在计算第 {i+1}/{total_professors} 位教授的相似度..."
                    )
                    progress_container.info(f"🔄 分析中: {professor.get('Professor Name', 'Unknown')}")

                    # 计算相似度
                    similarity = calculate_advanced_similarity(
                        professor.get("Research Interests", ""),
                        professor.get("Keywords", []),
                        user_interests,
                        st.session_state.api_key,
                    )
                    # 提取相似度分数
                    similarity_data = extract_structured_similarity_data(similarity)
                    
                    # 验证分数一致性
                    if not validate_similarity_scores(similarity_data):
                        st.warning(f"⚠️ {professor.get('Professor Name', 'Unknown')} 的相似度分数可能不准确")

                    results.append(
                        {
                            "URL": professor.get("URL", ""),
                            "Professor Name": professor.get("Professor Name", ""),
                            "Title": professor.get("Title", ""),
                            "Department": professor.get("Department", ""),
                            "Research Interests": professor.get("Research Interests", ""),
                            "Keywords": professor.get("Keywords", []),
                            "Additional URLs": professor.get("Additional URLs", []),
                            "Confidence Score": professor.get("Confidence Score", 0),
                            "Similarity Analysis": similarity,
                            "Score": similarity_data['overall_score'],
                            "Dimension Scores": similarity_data.get('dimension_scores', {}),
                            "Matched Keywords": similarity_data.get('matched_keywords', []),
                            "Similarity Reasoning": similarity_data.get('reasoning', {}),
                            "Analysis Confidence": similarity_data.get('confidence', 0.0),
                            "Parsing Success": similarity_data.get('success', False)
                        }
                    )

                # 完成所有计算
                progress_bar.progress(1.0)
                progress_text.text("✅ 相似度计算完成!")
                time.sleep(0.5)  # 短暂显示完成状态
                progress_bar.empty()
                progress_text.empty()
                progress_container.empty()

                # 保存结果到会话状态
                st.session_state.similarity_results = pd.DataFrame(results)

                # 显示结果
                display_advanced_similarity_results(st.session_state.similarity_results)

        # 处理排序按钮
        elif (
            sort_button and has_similarity_results
        ):
            try:
                # 确保数据是DataFrame格式
                if isinstance(st.session_state.similarity_results, pd.DataFrame):
                    sorted_results = st.session_state.similarity_results.sort_values(
                        by="Score", ascending=False
                    )
                else:
                    # 如果不是DataFrame，先转换
                    sorted_results = pd.DataFrame(st.session_state.similarity_results).sort_values(
                        by="Score", ascending=False
                    )

                st.subheader("📊 排序后的相似度分析结果")
                display_advanced_similarity_results(sorted_results)
            except Exception as e:
                st.error(f"排序失败: {str(e)}")
                st.write("尝试显示原始结果:")
                display_advanced_similarity_results(st.session_state.similarity_results)

        # 如果已有结果但未点击任何按钮，显示之前的结果
        elif has_similarity_results:
            st.subheader("📊 上次相似度分析结果")
            display_advanced_similarity_results(st.session_state.similarity_results)


if __name__ == "__main__":
    main()
