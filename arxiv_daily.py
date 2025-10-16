from llm import *
from util.request import get_yesterday_arxiv_papers
from util.construct_email import *
from tqdm import tqdm
import json
import os
from datetime import datetime, timezone
import time
import random
import smtplib
from email.header import Header
from email.utils import parseaddr, formataddr
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests
from rank_bm25 import BM25Okapi
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch



class ArxivDaily:
    def __init__(
        self,
        categories: list[str],
        max_entries: int,
        max_paper_num: int,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        description: str,
        num_workers: int,
        temperature: float,
        save_dir: str,
        filter_method: str = "bm25",  
        language: str = "korean",     
    ):
        self.model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_paper_num = max_paper_num
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.temperature = temperature
        self.filter_method = filter_method.lower()  
        self.language = language.lower()            

        self.run_datetime = datetime.now(timezone.utc)
        self.run_date = self.run_datetime.strftime("%Y-%m-%d")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(base_dir, save_dir, self.run_date,"json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.papers = {}
        for category in categories:
            self.papers[category] = get_yesterday_arxiv_papers(category, max_entries)
            print(
                "{} papers on arXiv for {} are fetched.".format(
                    len(self.papers[category]), category
                )
            )
            # avoid being blocked
            sleep_time = random.randint(5, 15)
            time.sleep(sleep_time)

        provider = provider.lower()
        if provider == "ollama":
            self.model = Ollama(model)
        elif provider == "openai" or provider == "siliconflow":
            self.model = GPT(model, base_url, api_key)
        else:
            assert False, "Model not supported."
        print(
            "Model initialized successfully. Using {} provided by {}.".format(
                model, provider
            )
        )

        self.description = description
        self.lock = threading.Lock()  # 添加线程锁

    def get_response(self, title, abstract):
        if self.language == "english":
            prompt = f"""
            You are a helpful academic research assistant.
            Below is a description of my current research area:
            {self.description}

            Here is a paper retrieved from arXiv yesterday.
            Title: {title}
            Abstract: {abstract}

            Please do the following:
            1. Summarize the main content of this paper in English (max 3 sentences).
            2. Evaluate how relevant it is to my research field (0–10, where 0 = not related, 10 = highly relevant).

            Return exactly in the following JSON format:
            {{
                "summary": "<summary>",
                "relevance": <score>
            }}

            ⚠️ Important:
            - Use English only.
            - Return JSON only, with no explanations or code blocks.
            """
        
        elif self.language == "chinese":
            prompt = """
                你是一个有帮助的学术研究助手，可以帮助我构建每日论文推荐系统。
                以下是我最近研究领域的描述：
                {}
            """.format(self.description)
            prompt += """
                以下是我从昨天的 arXiv 爬取的论文，我为你提供了标题和摘要：
                标题: {}
                摘要: {}
            """.format(title, abstract)
            prompt += """
                1. 总结这篇论文的主要内容。
                2. 请评估这篇论文与我研究领域的相关性，并给出 0-10 的评分。其中 0 表示完全不相关，10 表示高度相关。
                
                请按以下 JSON 格式给出你的回答：
                {
                    "summary": <你的总结>,
                    "relevance": <你的评分>
                }
                使用中文回答。
                直接返回上述 JSON 格式，无需任何额外解释。
            """

        else:  # default = korean
            prompt = f"""
            당신은 도움이 되는 학문 연구 어시스턴트입니다.
            아래는 내가 연구 중인 분야 설명입니다:
            {self.description}

            다음은 어제 arXiv에서 수집한 논문입니다.
            제목: {title}
            초록: {abstract}

            아래 두 가지를 수행하세요:
            1. 이 논문의 핵심 내용을 한국어로 간결하게 요약하세요. (최대 3문장)
            2. 내 연구 분야와의 관련도를 0~10점으로 평가하세요. (0 = 전혀 무관, 10 = 매우 관련)

            다음 JSON 형식으로 정확히 출력하세요:
            {{
                "summary": "<요약문>",
                "relevance": <점수>
            }}

            ⚠️주의:
            - 반드시 한국어로 작성하세요.
            - JSON 외의 불필요한 문장이나 코드블록(```json 등) 없이 결과만 출력하세요.
            """

        response = self.model.inference(prompt, temperature=self.temperature)
        return response


    def bm25_filter(self, papers, description, top_k=50):
        corpus = [(p["title"] + " " + p["abstract"]).lower().split() for p in papers]
        bm25 = BM25Okapi(corpus)
        query = description.lower().split()
        scores = bm25.get_scores(query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        filtered = [papers[i] for i in ranked_idx]
        for i, paper in enumerate(filtered):
            paper["bm25_score"] = scores[ranked_idx[i]]
        return filtered


    def dpr_filter(self, papers, description, top_k=50):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hasattr(self, "_dpr_cache"):
            print(f"[DPR] Loading models on {device} ...")
            self._dpr_cache = {
                "q_tok": DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base"),
                "q_enc": DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device),
                "c_tok": DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base"),
                "c_enc": DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device),
            }
            self._dpr_cache["q_enc"].eval()
            self._dpr_cache["c_enc"].eval()

        q_tok, q_enc = self._dpr_cache["q_tok"], self._dpr_cache["q_enc"]
        c_tok, c_enc = self._dpr_cache["c_tok"], self._dpr_cache["c_enc"]

        q_inputs = q_tok(description, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            q_emb = q_enc(**q_inputs).pooler_output

        doc_texts = [p["title"] + " " + p["abstract"] for p in papers]
        doc_embs = []
        for text in doc_texts:
            d_inputs = c_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                d_emb = c_enc(**d_inputs).pooler_output
            doc_embs.append(d_emb.cpu())

        doc_embs = torch.cat(doc_embs, dim=0)
        sims = torch.matmul(q_emb.cpu(), doc_embs.T).squeeze(0)
        top_idx = torch.topk(sims, k=min(top_k, len(sims))).indices.tolist()
        filtered = [papers[i] for i in top_idx]

        for i, p in enumerate(filtered):
            p["dpr_score"] = float(sims[top_idx[i]])
        return filtered

    
    def splade_filter(self, papers, description, top_k=50):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hasattr(self, "_splade_cache"):
            print(f"[Splade] Loading model on {device} ...")
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            tok = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
            model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
            model.eval()
            self._splade_cache = {"tok": tok, "model": model}

        tok, model = self._splade_cache["tok"], self._splade_cache["model"]

        def encode(text):
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.squeeze(0)
            weights = torch.log(1 + torch.relu(logits))
            vec = torch.max(weights, dim=0).values
            return vec.cpu()

        q_vec = encode(description)
        doc_vecs = [encode(p["title"] + " " + p["abstract"]) for p in papers]

        sims = []
        for v in doc_vecs:
            denom = (torch.norm(q_vec) * torch.norm(v)).item()
            sims.append((torch.dot(q_vec, v) / denom).item() if denom != 0 else 0.0)

        sorted_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        filtered = [papers[i] for i in sorted_idx[:top_k]]

        for i, p in enumerate(filtered):
            p["splade_score"] = float(sims[sorted_idx[i]])
        return filtered



    def process_paper(self, paper, max_retries=5):
        retry_count = 0
        cache_path = os.path.join(self.cache_dir, f"{paper['arXiv_id']}.json")

        # Try to load from cache if available
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as cache_file:
                    cached_result = json.load(cache_file)
                print(f"✅ Successfully loaded cache file: {cache_path}")
                return cached_result
            except (json.JSONDecodeError, OSError) as e:
                print(f"⚠️ Failed to load cache file {cache_path}: {e} → Rebuilding new file.")

        # Try processing the paper within the retry limit
        while retry_count < max_retries:
            try:
                title = paper["title"]
                abstract = paper["abstract"]
                response = self.get_response(title, abstract)
                response = response.strip("```").strip("json")
                response = json.loads(response)

                relevance_score = float(response["relevance"])
                summary = response["summary"]

                result = {
                    "title": title,
                    "arXiv_id": paper["arXiv_id"],
                    "abstract": abstract,
                    "summary": summary,
                    "relevance_score": relevance_score,
                    "pdf_url": paper["pdf_url"],
                }

                # Save to cache
                try:
                    with self.lock:
                        with open(cache_path, "w", encoding="utf-8") as cache_file:
                            json.dump(result, cache_file, ensure_ascii=False, indent=2)
                    print(f"💾 Cache saved successfully: {cache_path}")
                except OSError as write_error:
                    print(f"⚠️ Error occurred while saving cache: {write_error}")

                return result

            except Exception as e:
                retry_count += 1
                print(f"❌ Error while processing paper {paper['arXiv_id']}: {e}")
                print(f"⏳ Retrying... ({retry_count}/{max_retries})")

                if retry_count == max_retries:
                    print(f"🚫 Exceeded maximum retries ({max_retries}). Skipping paper {paper['arXiv_id']}.")
                    # Return fallback result on failure
                    result = {
                        "title": paper["title"],
                        "arXiv_id": paper["arXiv_id"],
                        "abstract": paper["abstract"],
                        "summary": "Failed to summarize this paper.",
                        "relevance_score": 0.0,
                        "pdf_url": paper.get("pdf_url", ""),
                    }
                    try:
                        with self.lock:
                            with open(cache_path, "w", encoding="utf-8") as cache_file:
                                json.dump(result, cache_file, ensure_ascii=False, indent=2)
                        print(f"⚠️ Saved failure result to cache: {cache_path}")
                    except OSError as write_error:
                        print(f"⚠️ Error occurred while saving failure result: {write_error}")
                    return result

                time.sleep(1)  # Wait before retrying



    def get_recommendation(self):
        recommendations = {}

        for category, papers in self.papers.items():
            # Use the filter method chosen by the user (from --filter_method)
            if self.filter_method == "bm25":
                filtered = self.bm25_filter(papers, self.description, top_k=self.max_paper_num * 2)
            elif self.filter_method == "dpr":
                filtered = self.dpr_filter(papers, self.description, top_k=self.max_paper_num * 2)
            elif self.filter_method == "splade":
                filtered = self.splade_filter(papers, self.description, self.max_paper_num * 2)
            else:
                filtered = papers  # No filtering

            for p in filtered:
                recommendations[p["arXiv_id"]] = p

        print(f"[Filter: {self.filter_method.upper()}] Retrieved {len(recommendations)} papers after filtering.")
    
        # Proceed to LLM inference
        recommendations_ = []
        print("Performing LLM inference...")

        with ThreadPoolExecutor(self.num_workers) as executor:
            futures = []
            for arXiv_id, paper in recommendations.items():
                futures.append(executor.submit(self.process_paper, paper))
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing papers",
                unit="paper",
            ):
                result = future.result()
                if result:
                    recommendations_.append(result)

        # Sort by LLM relevance score
        recommendations_ = sorted(
            recommendations_, key=lambda x: x["relevance_score"], reverse=True
        )[: self.max_paper_num]

        # Save recommendation to markdown file
        current_time = self.run_datetime
        save_path = os.path.join(
            self.save_dir, self.run_date, f"{current_time.strftime('%Y-%m-%d')}.md"
        )
        with open(save_path, "w") as f:
            f.write("# Daily arXiv Papers\n")
            f.write(f"## Date: {current_time.strftime('%Y-%m-%d')}\n")
            f.write(f"## Description: {self.description}\n")
            f.write("## Papers:\n")
            for i, paper in enumerate(recommendations_):
                f.write(f"### {i + 1}. {paper['title']}\n")
                f.write(f"#### Abstract:\n")
                f.write(f"{paper['abstract']}\n")
                f.write(f"#### Summary:\n")
                f.write(f"{paper['summary']}\n")
                f.write(f"#### Relevance Score: {paper['relevance_score']}\n")
                f.write(f"#### PDF URL: {paper['pdf_url']}\n")
                f.write("\n")

        return recommendations_

    def summarize(self, recommendations):
        overview = ""
        for i in range(len(recommendations)):
            overview += f"{i + 1}. {recommendations[i]['title']} - {recommendations[i]['summary']}\n"

        # -------- Chinese Version --------
        if self.language == "chinese":

            prompt_context = """
                你是一个有帮助的学术研究助手，可以帮助我构建每日论文推荐系统。
                以下是我最近研究领域的描述：
                {}
            """.format(self.description)
            papers_context = """
                以下是我从昨天的 arXiv 爬取的论文，我为你提供了标题和摘要：
                {}
            """.format(overview)
            json_instruction = """
                请务必严格按照以下 JSON 结构返回内容，不要添加额外文本或代码块：
                {{
                "trend_summary": "<总体趋势，用中文,使用 html 的语法，不要使用 markdown 的语法>",
                "recommendations": [
                    {{
                    "title": "<论文标题>",
                    "relevance_label": "<高度相关/相关/一般相关>",
                    "recommend_reason": "<为什么值得我读>",
                    "key_contribution": "<一句话概括论文关键贡献>"
                    }}
                ],
                "additional_observation": "<补充观察，若无请写‘暂无’>"
                }}

                任务要求：
                1. 给出今天论文体现的整体研究趋势，解释其与我研究兴趣的联系。
                2. 精选最值得我精读的论文（建议返回 3-5 篇，可按实际情况增减），说明推荐理由并突出关键贡献。
                3. 如有需要持续关注或潜在风险的方向，请在补充观察中说明；若没有请写“暂无”。
            """
            html_instruction = """
                请直接输出一段 HTML 片段，严格遵循以下结构，不要包含 JSON、Markdown 或多余说明：
                <div class="summary-wrapper">
                <div class="summary-section">
                    <h2>今日研究趋势</h2>
                    <p>...</p>
                </div>
                <div class="summary-section">
                    <h2>重点推荐</h2>
                    <ol class="summary-list">
                    <li class="summary-item">
                        <div class="summary-item__header"><span class="summary-item__title">论文标题</span><span class="summary-pill">相关性</span></div>
                        <p><strong>推荐理由：</strong>...</p>
                        <p><strong>关键贡献：</strong>...</p>
                    </li>
                    </ol>
                </div>
                <div class="summary-section">
                    <h2>补充观察</h2>
                    <p>暂无或其他补充。</p>
                </div>
                </div>

                HTML 要用中文撰写内容，重点推荐部分建议返回 3-5 篇论文，可按实际情况增减，缺少推荐时请写“暂无推荐。”。
            """

             # -------- English Version --------
        elif self.language == "english":
            prompt_context = """
                You are a helpful academic research assistant.
                Below is the description of my current research field:
                {}
            """.format(self.description)
            papers_context = """
                Here are papers collected from arXiv yesterday.
                Each paper includes its title and summary:
                {}
            """.format(overview)
            json_instruction = """
                Please strictly follow the JSON format below without adding extra text or code blocks:
                {{
                "trend_summary": "<Overall research trend in English, using HTML syntax (no Markdown)>",
                "recommendations": [
                    {{
                    "title": "<Paper title>",
                    "relevance_label": "<Highly relevant / Relevant / Moderately relevant>",
                    "recommend_reason": "<Why this paper is worth reading>",
                    "key_contribution": "<One-sentence key contribution>"
                    }}
                ],
                "additional_observation": "<Additional remarks — 'None' if no extra notes>"
                }}

                Task requirements:
                1. Summarize the overall research trend shown in today’s papers and its relation to my research interest.
                2. Select 3–5 most valuable papers and explain why they are recommended.
                3. Provide any additional observations or future directions if relevant.
            """
            html_instruction = """
                Please output a pure HTML snippet following the structure below.
                Do NOT include JSON, Markdown, or any explanation text.
                <div class="summary-wrapper">
                <div class="summary-section">
                    <h2>Today's Research Trend</h2>
                    <p>...</p>
                </div>
                <div class="summary-section">
                    <h2>Top Recommendations</h2>
                    <ol class="summary-list">
                    <li class="summary-item">
                        <div class="summary-item__header">
                        <span class="summary-item__title">Paper title</span>
                        <span class="summary-pill">Relevance</span>
                        </div>
                        <p><strong>Reason:</strong> ...</p>
                        <p><strong>Key contribution:</strong> ...</p>
                    </li>
                    </ol>
                </div>
                <div class="summary-section">
                    <h2>Additional Observations</h2>
                    <p>None</p>
                </div>
                </div>
            """

        # -------- Korean Version (Default) --------
        else:
            prompt_context = """
                당신은 유능한 학문 연구 어시스턴트로서,
                매일 arXiv 논문을 분석하고 연구자에게 유용한 요약을 제공합니다.
                아래는 나의 최근 연구 관심 분야입니다:
                {}
            """.format(self.description)
            papers_context = """
                아래는 어제 arXiv에서 수집한 논문 목록입니다.
                각 논문의 제목과 요약이 제공됩니다:
                {}
            """.format(overview)
            json_instruction = """
                아래 JSON 형식을 **엄격히** 지켜서 응답하세요.
                추가 텍스트나 코드블록, 주석은 포함하지 마세요:
                {{
                "trend_summary": "<전체적인 연구 동향 — 한국어로, HTML 문법 사용 가능>",
                "recommendations": [
                    {{
                    "title": "<논문 제목>",
                    "relevance_label": "<매우 관련 / 관련 / 보통 관련>",
                    "recommend_reason": "<이 논문을 추천하는 이유>",
                    "key_contribution": "<논문의 핵심 기여>"
                    }}
                ],
                "additional_observation": "<추가 관찰 내용 — 없으면 '없음'>"
                }}
            """
            html_instruction = """
                아래 예시 HTML 구조를 그대로 따르세요.
                JSON, Markdown, 설명문 없이 **순수 HTML**만 반환해야 합니다:
                <div class="summary-wrapper">
                <div class="summary-section">
                    <h2>오늘의 연구 동향</h2>
                    <p>...</p>
                </div>
                <div class="summary-section">
                    <h2>주요 추천</h2>
                    <ol class="summary-list">
                    <li class="summary-item">
                        <div class="summary-item__header">
                        <span class="summary-item__title">논문 제목</span>
                        <span class="summary-pill">관련도</span>
                        </div>
                        <p><strong>추천 이유:</strong> ...</p>
                        <p><strong>핵심 기여:</strong> ...</p>
                    </li>
                    </ol>
                </div>
                <div class="summary-section">
                    <h2>보충 관찰</h2>
                    <p>없음</p>
                </div>
                </div>
            """

        # Common parts shared by all languages
        prompt = prompt_context + papers_context + json_instruction
        html_prompt = prompt_context + papers_context + html_instruction

        def _clean_model_response(raw_text: str) -> str:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                if "\n" in cleaned:
                    first_line, rest = cleaned.split("\n", 1)
                    if first_line.strip().lower() in ("json", "html"):
                        cleaned = rest
                    else:
                        cleaned = first_line + "\n" + rest
            return cleaned.strip()

        max_retries = 1
        for attempt in range(1, max_retries + 1):
            try:
                raw_response = self.model.inference(prompt, temperature=self.temperature)
                cleaned = _clean_model_response(raw_response)
                data = json.loads(cleaned)

                if self.language == "english":
                    no_trend = "No trend information available."
                    no_observation = "None."
                    relevance_unknown = "Unknown relevance"
                    no_reason = "No recommendation reason provided."
                    no_contribution = "No key contribution provided."
                    field_error = "The 'recommendations' field must be a list."
                    missing_title = "A recommendation entry is missing its title."
                    summary_fail = "Summary generation failed. Please try again later."
                    html_retry_msg = lambda n: f"HTML fallback attempt {n}..."
                    summary_retry_msg = lambda n, e: f"Summary generation attempt {n} failed: {e}"

                elif self.language == "chinese":
                    no_trend = "暂无趋势信息"
                    no_observation = "暂无"
                    relevance_unknown = "相关性未知"
                    no_reason = "未提供推荐理由"
                    no_contribution = "未提供关键贡献"
                    field_error = "recommendations 字段不是列表"
                    missing_title = "recommendations 中存在缺少标题的条目"
                    summary_fail = "总结生成失败，请稍后重试。"
                    html_retry_msg = lambda n: f"HTML 回退生成第 {n} 次..."
                    summary_retry_msg = lambda n, e: f"总结生成第 {n} 次失败: {e}"

                else:  # Default: Korean
                    no_trend = "연구 동향 정보 없음"
                    no_observation = "없음"
                    relevance_unknown = "관련도 불명"
                    no_reason = "추천 이유가 없습니다."
                    no_contribution = "핵심 기여가 제공되지 않았습니다."
                    field_error = "recommendations 필드는 리스트여야 합니다."
                    missing_title = "추천 항목에 제목이 없습니다."
                    summary_fail = "요약 생성에 실패했습니다. 나중에 다시 시도해주세요."
                    html_retry_msg = lambda n: f"HTML 대체 생성 시도 {n}번째..."
                    summary_retry_msg = lambda n, e: f"요약 생성 {n}번째 실패: {e}"

                trend_summary = data.get("trend_summary", no_trend)
                recommendations_data = data.get("recommendations", [])
                additional_observation = data.get("additional_observation", no_observation)

                if not isinstance(recommendations_data, list):
                    raise ValueError(field_error)

                cleaned_recommendations = []
                for item in recommendations_data:
                    title = item.get("title")
                    if not title:
                        raise ValueError(missing_title)
                    cleaned_recommendations.append(
                        {
                            "title": title,
                            "relevance_label": item.get("relevance_label", relevance_unknown),
                            "recommend_reason": item.get("recommend_reason", no_reason),
                            "key_contribution": item.get("key_contribution", no_contribution),
                        }
                    )

                structured_summary = {
                    "trend_summary": trend_summary,
                    "recommendations": cleaned_recommendations,
                    "additional_observation": additional_observation,
                }

                return render_summary_sections(structured_summary)

            except Exception as error:
                print(summary_retry_msg(attempt, error))
                if attempt == max_retries:
                    try:
                        for html_attempt in range(1, max_retries + 1):
                            print(html_retry_msg(html_attempt))
                            raw_html_response = self.model.inference(html_prompt, temperature=self.temperature)
                            cleaned_html = _clean_model_response(raw_html_response)
                            return cleaned_html
                    except Exception as html_error:
                        print(f"[HTML Fallback Error] {html_error}")
                        fallback_data = {
                            "trend_summary": summary_fail,
                            "recommendations": [],
                            "additional_observation": no_observation,
                        }
                        return render_summary_sections(fallback_data)



    def render_email(self, recommendations):
        # Set the path for saving the rendered HTML email
        save_file_path = os.path.join(self.save_dir, self.run_date, "arxiv_daily_email.html")

        # Load from cache if the email has already been rendered
        if os.path.exists(save_file_path):
            with open(save_file_path, "r", encoding="utf-8") as f:
                print(f"Email already rendered. Loading from cache file: {save_file_path}")
                return f.read()

        parts = []

        # Return empty HTML if no recommendations exist
        if len(recommendations) == 0:
            return framework.replace("__CONTENT__", get_empty_html())

        # Convert each paper's information into an HTML block
        for i, p in enumerate(tqdm(recommendations, desc="Rendering email")):
            rate = get_stars(p["relevance_score"])
            parts.append(
                get_block_html(
                    str(i + 1) + ". " + p["title"],  # title
                    rate,                            # star rating
                    p["arXiv_id"],                   # arXiv ID
                    p["summary"],                    # summary
                    p["pdf_url"],                    # PDF link
                )
            )

        # Add overall summary to the top of the email
        summary = self.summarize(recommendations)
        content = summary
        content += "<br>" + "</br><br>".join(parts) + "</br>"

        # Insert content into the HTML framework
        email_html = framework.replace("__CONTENT__", content)

        # Save the rendered email to the specified directory
        if self.save_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(base_dir, self.save_dir, self.run_date, "arxiv_daily_email.html")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(email_html)

        return email_html


    def send_email(
        self,
        sender: str,
        receiver: str,
        password: str,
        smtp_server: str,
        smtp_port: int,
        title: str,
    ):
        recommendations = self.get_recommendation()
        html = self.render_email(recommendations)

        def _format_addr(s):
            name, addr = parseaddr(s)
            return formataddr((Header(name, "utf-8").encode(), addr))

        msg = MIMEText(html, "html", "utf-8")
        msg["From"] = _format_addr(f"{title} <%s>" % sender)

        receivers = [addr.strip() for addr in receiver.split(",")]
        print(receivers)
        msg["To"] = ",".join([_format_addr(f"You <%s>" % addr) for addr in receivers])

        today = self.run_datetime.strftime("%Y/%m/%d")
        msg["Subject"] = Header(f"{title} {today}", "utf-8").encode()

        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
        except Exception as e:
            logger.warning(f"Failed to use TLS. {e}")
            logger.warning(f"Try to use SSL.")
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)

        server.login(sender, password)
        server.sendmail(sender, receivers, msg.as_string())
        server.quit()


    def send_slack(self, webhook_url: str, title: str):
        recommendations = self.get_recommendation()

        self.render_email(recommendations)

        # Select the top 3 papers based on relevance_score
        top_recommendations = sorted(
            recommendations, key=lambda x: x["relevance_score"], reverse=True
        )[:3]

        # Build Slack message blocks (Markdown-based formatting)
        message_blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"📚 {title} ({self.run_date})"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Here are today’s Top 3 paper summaries.*"}
            },
            {"type": "divider"}
        ]

        for i, paper in enumerate(top_recommendations, start=1):
            text = (
                f"*{i}. {paper['title']}*\n"
                f"{paper['summary'][:500]}...\n"
                f"<{paper['pdf_url']}|[PDF Link]>  ·  Relevance: {paper['relevance_score']:.1f}"
            )
            message_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})
            message_blocks.append({"type": "divider"})

        # Send message payload to Slack webhook
        payload = {"blocks": message_blocks}
        response = requests.post(webhook_url, json=payload)

        if response.status_code == 200:
            print("✅ Successfully sent Slack message (Top 3 papers).")
        else:
            print(f"⚠️ Failed to send Slack message: {response.status_code}, {response.text}")



if __name__ == "__main__":
    categories = ["cs.CV"]
    max_entries = 100
    max_paper_num = 50
    provider = "ollama"
    model = "deepseek-r1:7b"
    description = """
        My research focuses on building intelligent speech systems that combine language understanding and generation. 
        I am particularly interested in how speech, prosody, and emotion can be modeled together with language to create more natural and human-like AI agents. 
        My work explores generative and multimodal approaches that unify speech, text, and visual information — enabling AI systems that can not only speak but also understand context, emotion, and intent in conversation.
    """
    

    arxiv_daily = ArxivDaily(
        categories, max_entries, max_paper_num, provider, model, None, None, description
    )
    recommendations = arxiv_daily.get_recommendation()
    print(recommendations)
