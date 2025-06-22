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


def render_keyword_tag(keyword: str) -> str:
    """渲染主题自适应的关键词标签"""
    # 使用CSS变量和更好的颜色方案，支持明亮和夜间主题
    style = """
    background-color: var(--background-color, rgba(59, 130, 246, 0.1)); 
    color: var(--text-color, #1e40af); 
    border: 1px solid var(--border-color, rgba(59, 130, 246, 0.3));
    padding: 2px 8px; 
    margin: 2px; 
    border-radius: 12px; 
    display: inline-block;
    font-size: 0.85em;
    font-weight: 500;
    """
    
    # 添加CSS变量定义，适配不同主题
    css_vars = """
    <style>
    :root {
        --background-color: rgba(59, 130, 246, 0.1);
        --text-color: #1e40af;
        --border-color: rgba(59, 130, 246, 0.3);
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: rgba(59, 130, 246, 0.2);
            --text-color: #93c5fd;
            --border-color: rgba(59, 130, 246, 0.4);
        }
    }
    /* Streamlit夜间模式兼容 */
    .stApp[data-theme="dark"] {
        --background-color: rgba(59, 130, 246, 0.2);
        --text-color: #93c5fd;
        --border-color: rgba(59, 130, 246, 0.4);
    }
    </style>
    """
    
    return f"{css_vars}<span style='{style}'>{keyword}</span>"


def calculate_similarity(prof_interests, user_interests, api_key):
    """使用LLM计算教授研究兴趣与用户兴趣的相似度"""
    try:
        # 创建客户端
        current_client = get_client(api_key)

        llm_response = current_client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115",
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


def extract_similarity_score(similarity_text):
    """从相似度文本中提取分数"""
    try:
        # 尝试查找数字模式，例如"相似度：85分"或"相似度分数：85/100"
        matches = re.findall(r"(\d+)(?:/100|\s*分|\s*点)", similarity_text)
        if matches:
            return int(matches[0])

        # 如果没有找到格式化的分数，尝试查找单独的数字
        matches = re.findall(r"(?<!\d)\b(\d{1,3})\b(?!\d)", similarity_text)
        if matches:
            # 过滤掉不太可能是分数的数字（太小或太大）
            possible_scores = [int(m) for m in matches if 0 <= int(m) <= 100]
            if possible_scores:
                return max(possible_scores)  # 返回最可能的分数

        return 0  # 默认值
    except Exception:
        return 0  # 出错时返回默认值





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
        st.warning("⚠️ 没有结果可显示")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 分离教授页面和非教授页面
    professor_results = [r for r in results if r.get('Is Professor Page') == 'Yes']
    
    if professor_results:
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
            with st.expander(f"👨‍🏫 {result.get('Professor Name', 'Unknown')} - {result.get('Title', '')}"):
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
    else:
        st.warning("⚠️ 未找到教授页面")
    
    # 显示分析统计
    st.markdown("---")
    st.markdown("#### 📊 分析统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总链接数", len(results))
    with col2:
        st.metric("教授页面", len(professor_results))
    with col3:
        success_rate = len(professor_results) / len(results) * 100 if results else 0
        st.metric("成功率", f"{success_rate:.1f}%")
    with col4:
        avg_confidence = sum(r.get('Confidence Score', 0) for r in professor_results) / len(professor_results) if professor_results else 0
        st.metric("平均置信度", f"{avg_confidence:.1%}")


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
            professor_results = [r for r in st.session_state.results if r.get('Is Professor Page') == 'Yes']
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
            sort_button = st.button(
                "📊 排序结果",
                disabled="similarity_results" not in st.session_state
                or st.session_state.similarity_results is None,
                key="sort_results_button"
            )
            if (
                "similarity_results" not in st.session_state
                or st.session_state.similarity_results is None
            ):
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
                    progress_container.info(f"🔄 分析中: {professor.get('url', 'Unknown')}")

                    # 计算相似度
                    similarity = calculate_similarity(
                        professor.get("research_interests", ""),
                        user_interests,
                        st.session_state.api_key,
                    )
                    # 提取相似度分数
                    similarity_score = extract_similarity_score(similarity)

                    results.append(
                        {
                            "URL": professor.get("url", ""),
                            "Professor Name": professor.get("professor_name", ""),
                            "Title": professor.get("title", ""),
                            "Department": professor.get("department", ""),
                            "Research Interests": professor.get("research_interests", ""),
                            "Keywords": professor.get("keywords", []),
                            "Additional URLs": professor.get("additional_urls", []),
                            "Confidence Score": professor.get("confidence_score", 0),
                            "Similarity Analysis": similarity,
                            "Score": similarity_score,
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
                display_results(st.session_state.similarity_results)

        # 处理排序按钮
        elif (
            sort_button
            and "similarity_results" in st.session_state
            and st.session_state.similarity_results is not None
        ):
            # 按相似度分数排序结果
            sorted_results = st.session_state.similarity_results.sort_values(
                by="Score", ascending=False
            )

            st.subheader("📊 排序后的相似度分析结果")
            display_professor_results(sorted_results.to_dict(orient='records'), key_prefix="tab2_sorted")

        # 如果已有结果但未点击任何按钮，显示之前的结果
        elif (
            "similarity_results" in st.session_state
            and st.session_state.similarity_results is not None
        ):
            st.subheader("📊 上次相似度分析结果")
            display_professor_results(st.session_state.similarity_results.to_dict(orient='records'), key_prefix="tab2_saved")


if __name__ == "__main__":
    main()
