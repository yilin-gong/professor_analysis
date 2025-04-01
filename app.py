import streamlit as st
import pandas as pd
from main import analyze_webpage_links, get_all_links, is_professor_webpage, get_research_interests, get_client
import os
import tempfile
import concurrent.futures
import time
import requests
import re
from bs4 import BeautifulSoup

st.set_page_config(page_title="教授研究兴趣分析器 🎓", layout="wide")

# 初始化会话状态变量
if 'professor_results' not in st.session_state:
    st.session_state.professor_results = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'similarity_results' not in st.session_state:
    st.session_state.similarity_results = None

def calculate_similarity(prof_interests, user_interests, api_key):
    """使用LLM计算教授研究兴趣与用户兴趣的相似度"""
    try:
        # 创建客户端
        current_client = get_client(api_key)
            
        llm_response = current_client.chat.completions.create(
            model="doubao-1-5-pro-32k-250115",
            messages=[
                {"role": "system", "content": "你是一个专业的学术匹配专家。你的任务是分析教授的研究兴趣与用户的研究兴趣之间的相似度，给出0-100的分数和详细解释。"},
                {"role": "user", "content": f"请分析以下教授的研究兴趣与用户的研究兴趣之间的相似度。给出0-100的相似度分数和详细解释。\n\n教授研究兴趣: {prof_interests}\n\n用户研究兴趣: {user_interests}"}
            ]
        )
        return llm_response.choices[0].message.content.strip()
    except Exception as e:
        return f"分析相似度时出错: {str(e)}"

def extract_similarity_score(similarity_text):
    """从相似度文本中提取分数"""
    try:
        # 尝试查找数字模式，例如"相似度：85分"或"相似度分数：85/100"
        matches = re.findall(r'(\d+)(?:/100|\s*分|\s*点)', similarity_text)
        if matches:
            return int(matches[0])
        
        # 如果没有找到格式化的分数，尝试查找单独的数字
        matches = re.findall(r'(?<!\d)\b(\d{1,3})\b(?!\d)', similarity_text)
        if matches:
            # 过滤掉不太可能是分数的数字（太小或太大）
            possible_scores = [int(m) for m in matches if 0 <= int(m) <= 100]
            if possible_scores:
                return max(possible_scores)  # 返回最可能的分数
        
        return 0  # 默认值
    except Exception:
        return 0  # 出错时返回默认值

def analyze_with_progress(start_url, api_key, max_links=30, max_workers=5):
    """在界面上显示分析进度的函数"""
    if not api_key:
        st.error("❌ 请先在侧边栏设置API密钥")
        return pd.DataFrame(columns=["URL", "Is Professor Page", "Research Interests"])
    
    # 创建OpenAI客户端
    client = get_client(api_key)
    
    # 创建进度显示组件
    progress_container = st.empty()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # 步骤1: 获取所有链接
    progress_text.text("第1步: 🔍 获取链接中...")
    links = get_all_links(start_url)[:max_links]
    if not links:
        progress_text.text("❌ 无法获取任何链接，请检查URL是否正确")
        return pd.DataFrame(columns=["URL", "Is Professor Page", "Research Interests"])
    
    progress_container.info(f"🔗 共找到 {len(links)} 个链接，准备分析")
    
    # 步骤2: 分析每个链接
    progress_text.text("第2步: 🔎 分析链接中...")
    results = []
    session = requests.Session()
    
    # 使用线程池处理链接
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_link = {executor.submit(process_link_with_feedback, link, session, client): link for link in links}
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_link):
            link = future_to_link[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    # 更新状态信息
                    if result["Is Professor Page"] == "Yes":
                        progress_container.success(f"✅ 找到教授页面: {link}")
            except Exception as e:
                progress_container.error(f"❌ 处理 {link} 时出错: {str(e)}")
                results.append({
                    "URL": link,
                    "Is Professor Page": "Error",
                    "Research Interests": f"Error: {str(e)}"
                })
            
            # 更新进度条
            completed += 1
            progress = completed / len(links)
            progress_bar.progress(progress)
            progress_text.text(f"第2步: 🔎 分析链接中... ({completed}/{len(links)})")
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 分析完成
    professor_count = sum(1 for r in results if r["Is Professor Page"] == "Yes")
    progress_container.success(f"🎉 分析完成! 共找到 {professor_count} 个教授页面 (总共分析了 {len(results)} 个链接)")
    progress_text.empty()
    progress_bar.empty()
    
    # 保存到会话状态
    st.session_state.analysis_results = df
    # 提取教授页面
    st.session_state.professor_results = df[df["Is Professor Page"] == "Yes"]
    
    return df

def process_link_with_feedback(link, session, client):
    """处理单个链接，用于带进度显示的版本"""
    try:
        is_professor = is_professor_webpage(link, session, client)
        if is_professor:
            research_interests = get_research_interests(link, session, client)
            return {
                "URL": link,
                "Is Professor Page": "Yes",
                "Research Interests": research_interests
            }
        else:
            return {
                "URL": link,
                "Is Professor Page": "No",
                "Research Interests": ""
            }
    except Exception as e:
        return {
            "URL": link,
            "Is Professor Page": "Error",
            "Research Interests": f"Error: {str(e)}"
        }

# 添加一个函数来显示结果，避免代码重复
def display_results(results_df):
    """显示相似度结果"""
    if results_df is None or len(results_df) == 0:
        st.warning("⚠️ 没有结果可显示")
        return
        
    for _, row in results_df.iterrows():
        with st.expander(f"🔗 相似度 {row['Score']}分 - URL: {row['URL']}"):
            st.markdown("#### 👨‍🏫 教授研究兴趣")
            st.write(row["Research Interests"])
            st.markdown("#### 📊 相似度分析")
            st.write(row["Similarity Analysis"])

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
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            start_url = st.text_input("输入起始URL", "https://journalism.uiowa.edu/people")
        
        with col2:
            max_links = st.number_input("最大分析链接数", min_value=10, max_value=500, value=30, step=10)
            max_workers = st.number_input("工作线程数", min_value=1, max_value=10, value=5, step=1)
        
        if st.button("🚀 开始分析"):
            if not st.session_state.api_key:
                st.error("❌ 请先在侧边栏设置API密钥")
            else:
                # 使用包含进度显示的分析函数
                results_df = analyze_with_progress(start_url, st.session_state.api_key, max_links=max_links, max_workers=max_workers)
                
                # 保存结果到临时文件
                temp_dir = tempfile.mkdtemp()
                csv_path = os.path.join(temp_dir, "professor_analysis_results.csv")
                results_df.to_csv(csv_path, index=False)
                
                # 显示所有结果
                st.subheader("📋 分析结果")
                st.dataframe(results_df)
                
                # 显示教授页面
                professor_pages = results_df[results_df["Is Professor Page"] == "Yes"]
                if not professor_pages.empty:
                    st.subheader("👨‍🏫 找到的教授页面")
                    st.dataframe(professor_pages)
                    
                    # 提供下载链接
                    with open(csv_path, "rb") as file:
                        st.download_button(
                            label="📥 下载分析结果",
                            data=file,
                            file_name="professor_analysis_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.warning("⚠️ 未找到教授页面")
        
        # 显示会话中保存的结果
        elif st.session_state.analysis_results is not None:
            st.subheader("📋 上次分析结果")
            st.dataframe(st.session_state.analysis_results)
            
            if st.session_state.professor_results is not None and not st.session_state.professor_results.empty:
                st.subheader("👨‍🏫 找到的教授页面")
                st.dataframe(st.session_state.professor_results)
                
                # 创建临时文件供下载
                temp_dir = tempfile.mkdtemp()
                csv_path = os.path.join(temp_dir, "professor_analysis_results.csv")
                st.session_state.analysis_results.to_csv(csv_path, index=False)
                
                with open(csv_path, "rb") as file:
                    st.download_button(
                        label="📥 下载分析结果",
                        data=file,
                        file_name="professor_analysis_results.csv",
                        mime="text/csv"
                    )
    
    with tab2:
        st.header("🔄 研究兴趣相似度匹配")
        
        # 检查是否有教授结果
        professor_pages = None
        if st.session_state.professor_results is not None and not st.session_state.professor_results.empty:
            professor_pages = st.session_state.professor_results
            has_professors = True
        else:
            st.info("ℹ️ 请先在'🔍 网站分析'标签中分析一个网站以获取教授研究兴趣")
            professor_pages = pd.DataFrame(columns=["URL", "Is Professor Page", "Research Interests"])
            has_professors = False
        
        # 用户输入研究兴趣
        user_interests = st.text_area("✏️ 输入您的研究兴趣（请详细描述您感兴趣的研究领域、方法和主题）", height=150)
        has_interests = bool(user_interests.strip())
        
        # 始终显示按钮，但根据条件禁用
        col1, col2 = st.columns([3, 1])
        with col1:
            calc_button = st.button("🔍 计算相似度", disabled=not has_professors or not has_interests)
            if not has_professors and not has_interests:
                st.caption("请先分析网站并输入研究兴趣")
            elif not has_professors:
                st.caption("请先分析网站以获取教授数据")
            elif not has_interests:
                st.caption("请输入您的研究兴趣")
        with col2:
            sort_button = st.button("📊 排序结果", disabled="similarity_results" not in st.session_state or st.session_state.similarity_results is None)
            if "similarity_results" not in st.session_state or st.session_state.similarity_results is None:
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
                total_professors = len(professor_pages)
                
                for i, (_, row) in enumerate(professor_pages.iterrows()):
                    # 更新进度
                    progress = (i) / total_professors
                    progress_bar.progress(progress)
                    progress_text.text(f"⏳ 正在计算第 {i+1}/{total_professors} 位教授的相似度...")
                    progress_container.info(f"🔄 分析中: {row['URL']}")
                    
                    # 计算相似度
                    similarity = calculate_similarity(row["Research Interests"], user_interests, st.session_state.api_key)
                    # 提取相似度分数
                    similarity_score = extract_similarity_score(similarity)
                    
                    results.append({
                        "URL": row["URL"],
                        "Research Interests": row["Research Interests"],
                        "Similarity Analysis": similarity,
                        "Score": similarity_score
                    })
                
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
        elif sort_button and "similarity_results" in st.session_state and st.session_state.similarity_results is not None:
            # 按相似度分数排序结果
            sorted_results = st.session_state.similarity_results.sort_values(by="Score", ascending=False)
            
            st.subheader("📊 排序后的相似度分析结果")
            display_results(sorted_results)
            
        # 如果已有结果但未点击任何按钮，显示之前的结果
        elif "similarity_results" in st.session_state and st.session_state.similarity_results is not None:
            st.subheader("📊 上次相似度分析结果")
            display_results(st.session_state.similarity_results)

if __name__ == "__main__":
    main() 