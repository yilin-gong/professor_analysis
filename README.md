# 教授研究兴趣分析器

这是一个用于分析大学网站，查找教授页面并提取研究兴趣的工具，同时提供兴趣匹配功能。

## 功能

1. **网站分析**：自动爬取大学网站，识别教授页面并提取其研究兴趣
2. **兴趣匹配**：将您的研究兴趣与找到的教授研究兴趣进行比较，计算相似度分数
3. **翻页检测**：自动识别并跟随“下一页”链接，获取更多页面链接

## 安装

1. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```

2. 设置API密钥（可选，默认使用代码中的密钥）：
   ```python
   # 在main.py中修改
   api_key = '您的API密钥'  # 替换为您的API密钥
   ```

## 使用方法

1. 启动Web应用：
   ```
   streamlit run app.py
   ```

2. 在浏览器中使用应用：
   - **网站分析**标签：输入大学网站URL进行分析
   - **兴趣匹配**标签：输入您的研究兴趣，与找到的教授兴趣进行匹配

## 命令行使用

您也可以使用命令行版本：

```
python main.py 网站URL [--max-links 链接数量] [--max-pages 翻页数量] [--workers 线程数] [--output 输出文件名]
```

例如：
```
python main.py https://journalism.uiowa.edu/people --max-links 500 --max-pages 3 --workers 5
```
