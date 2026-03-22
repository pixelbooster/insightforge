"""
InsightForge — LangChain RAG Pipeline
======================================
Covers capstone Steps 3–6:
  Step 3c  : Custom LangChain BaseRetriever
  Step 3d  : Prompt engineering via PromptTemplate
  Step 4   : Chain prompts — LLMChain + SequentialChain
  Step 5   : RAG system — FAISS vector store + RetrievalQA
  Step 6   : Memory — ConversationBufferMemory
"""
import streamlit as st
import os
import pandas as pd
from typing import List, Any

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, LLMChain, SequentialChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# STEP 2  Knowledge base — build LangChain Documents from sales data
# ─────────────────────────────────────────────────────────────────
def build_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert aggregated pandas statistics into LangChain Documents.
    These become the knowledge base chunks stored in FAISS.
    """
    docs: List[Document] = []

    # ── Overall summary ─────────────────────────────────────────
    docs.append(Document(
        page_content=(
            f"Overall Business Summary\n"
            f"Total Records: {len(df):,}\n"
            f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}\n"
            f"Total Revenue: ${df['Sales'].sum():,.0f}\n"
            f"Average Sale: ${df['Sales'].mean():.2f}\n"
            f"Median Sale: ${df['Sales'].median():.2f}\n"
            f"Standard Deviation: ${df['Sales'].std():.2f}\n"
            f"Min Sale: ${df['Sales'].min()}\n"
            f"Max Sale: ${df['Sales'].max()}\n"
            f"Avg Customer Satisfaction: {df['Customer_Satisfaction'].mean():.3f}/5.0\n"
            f"Avg Customer Age: {df['Customer_Age'].mean():.1f} years\n"
            f"Best Product by Revenue: Widget A ($375,235)\n"
            f"Best Region by Revenue: West ($361,383)\n"
            f"Peak Year: 2026 ($206,175)\n"
            f"Peak Month: August (avg $572.6 per sale)\n"
            f"Trough Month: September (avg $531.5 per sale)"
        ),
        metadata={"category": "overview", "name": "overall"}
    ))

    # ── Per-product documents ────────────────────────────────────
    for product, grp in df.groupby("Product"):
        top_region = grp.groupby("Region")["Sales"].sum().idxmax()
        docs.append(Document(
            page_content=(
                f"Product Analysis: {product}\n"
                f"Total Revenue: ${grp['Sales'].sum():,.0f}\n"
                f"Average Sale Value: ${grp['Sales'].mean():.2f}\n"
                f"Number of Transactions: {len(grp):,}\n"
                f"Average Customer Satisfaction: {grp['Customer_Satisfaction'].mean():.3f}/5.0\n"
                f"Revenue Std Dev: ${grp['Sales'].std():.2f}\n"
                f"Best Performing Region: {top_region}\n"
                f"Male Customers Revenue: ${grp[grp['Customer_Gender']=='Male']['Sales'].sum():,.0f}\n"
                f"Female Customers Revenue: ${grp[grp['Customer_Gender']=='Female']['Sales'].sum():,.0f}"
            ),
            metadata={"category": "product", "name": product}
        ))

    # ── Per-region documents ─────────────────────────────────────
    for region, grp in df.groupby("Region"):
        top_product = grp.groupby("Product")["Sales"].sum().idxmax()
        share = grp["Sales"].sum() / df["Sales"].sum() * 100
        docs.append(Document(
            page_content=(
                f"Regional Analysis: {region}\n"
                f"Total Revenue: ${grp['Sales'].sum():,.0f}\n"
                f"Average Sale Value: ${grp['Sales'].mean():.2f}\n"
                f"Number of Transactions: {len(grp):,}\n"
                f"Market Share: {share:.1f}%\n"
                f"Best Performing Product: {top_product}\n"
                f"Avg Customer Age: {grp['Customer_Age'].mean():.1f} years\n"
                f"Avg Satisfaction: {grp['Customer_Satisfaction'].mean():.3f}/5.0"
            ),
            metadata={"category": "region", "name": region}
        ))

    # ── Annual revenue documents ─────────────────────────────────
    yr_rev = df.groupby("Year")["Sales"].sum()
    annual_text = "Annual Revenue Performance:\n"
    years_sorted = sorted(yr_rev.items())
    for i, (yr, rev) in enumerate(years_sorted):
        if i == 0:
            delta_str = "(baseline year)"
        else:
            prev = years_sorted[i - 1][1]
            pct = (rev - prev) / prev * 100
            delta_str = f"({pct:+.1f}% vs previous year)"
        annual_text += f"  {yr}: ${rev:,.0f} {delta_str}\n"
    docs.append(Document(
        page_content=annual_text,
        metadata={"category": "annual", "name": "annual_trend"}
    ))

    # ── Monthly seasonality document ─────────────────────────────
    month_avg = df.groupby("Month")["Sales"].mean().round(2)
    month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    monthly_text = "Monthly Sales Seasonality (average sale value):\n"
    for m, val in month_avg.items():
        flag = " ← PEAK" if val >= 570 else " ← TROUGH" if val <= 535 else ""
        monthly_text += f"  {month_names[m]}: ${val:.2f}{flag}\n"
    docs.append(Document(
        page_content=monthly_text,
        metadata={"category": "monthly", "name": "seasonality"}
    ))

    # ── Customer demographics document ──────────────────────────
    df_age = df.copy()
    df_age["AgeGroup"] = pd.cut(
        df_age["Customer_Age"],
        bins=[17, 30, 45, 60, 70],
        labels=["18-30", "31-45", "46-60", "61-70"]
    )
    demo_text = "Customer Demographics Analysis:\n\nBy Gender:\n"
    for gender, grp in df.groupby("Customer_Gender"):
        demo_text += (
            f"  {gender}: ${grp['Sales'].sum():,.0f} total | "
            f"${grp['Sales'].mean():.2f} avg | {len(grp):,} transactions\n"
        )
    demo_text += "\nBy Age Group:\n"
    for age_group, grp in df_age.groupby("AgeGroup", observed=True):
        demo_text += (
            f"  Age {age_group}: ${grp['Sales'].sum():,.0f} total | "
            f"${grp['Sales'].mean():.2f} avg | {len(grp):,} transactions\n"
        )
    docs.append(Document(
        page_content=demo_text,
        metadata={"category": "demographics", "name": "customer_segments"}
    ))

    # ── Product × Region combinations ────────────────────────────
    combos = df.groupby(["Product", "Region"])["Sales"].sum().reset_index()
    combos_text = "Top Product-Region Revenue Combinations:\n"
    for _, row in combos.sort_values("Sales", ascending=False).head(8).iterrows():
        combos_text += f"  {row['Product']} + {row['Region']}: ${row['Sales']:,.0f}\n"
    combos_text += "\nBottom Product-Region Combinations:\n"
    for _, row in combos.sort_values("Sales").head(4).iterrows():
        combos_text += f"  {row['Product']} + {row['Region']}: ${row['Sales']:,.0f}\n"
    docs.append(Document(
        page_content=combos_text,
        metadata={"category": "combinations", "name": "product_region"}
    ))

    # ── Satisfaction analysis ────────────────────────────────────
    sat_text = "Customer Satisfaction Analysis:\n\nBy Product:\n"
    for product, grp in df.groupby("Product"):
        sat_text += (
            f"  {product}: {grp['Customer_Satisfaction'].mean():.3f}/5.0 avg | "
            f"min {grp['Customer_Satisfaction'].min():.2f} | max {grp['Customer_Satisfaction'].max():.2f}\n"
        )
    sat_text += "\nBy Region:\n"
    for region, grp in df.groupby("Region"):
        sat_text += f"  {region}: {grp['Customer_Satisfaction'].mean():.3f}/5.0 avg\n"
    sat_text += (
        f"\nOverall: {df['Customer_Satisfaction'].mean():.3f}/5.0 — "
        f"Widget D has highest satisfaction (3.07), Widget B lowest (2.99)"
    )
    docs.append(Document(
        page_content=sat_text,
        metadata={"category": "satisfaction", "name": "satisfaction"}
    ))

    return docs


# ─────────────────────────────────────────────────────────────────
# STEP 3c  Custom LangChain BaseRetriever
# ─────────────────────────────────────────────────────────────────
class InsightRetriever(BaseRetriever):
    """
    Custom LangChain BaseRetriever.
    Wraps FAISS with intent-aware category filtering so the
    most relevant document categories are always retrieved.
    """
    vectorstore: Any
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_lower = query.lower()

        # Intent detection → preferred category
        preferred = None
        if any(w in query_lower for w in ["product", "widget", "item"]):
            preferred = "product"
        elif any(w in query_lower for w in ["region", "north", "south", "east", "west", "area"]):
            preferred = "region"
        elif any(w in query_lower for w in ["year", "annual", "trend", "growth", "decline"]):
            preferred = "annual"
        elif any(w in query_lower for w in ["month", "season", "january", "august", "quarterly"]):
            preferred = "monthly"
        elif any(w in query_lower for w in ["age", "demographic", "segment", "young", "senior"]):
            preferred = "demographics"
        elif any(w in query_lower for w in ["satisfaction", "happy", "rating", "score", "review"]):
            preferred = "satisfaction"
        elif any(w in query_lower for w in ["combo", "combination", "top", "best pair", "prioritize"]):
            preferred = "combinations"

        # Semantic similarity search (wider pool)
        candidates = self.vectorstore.similarity_search(query, k=self.k * 2)

        # Always include the overview document
        overview = [d for d in candidates if d.metadata.get("category") == "overview"]

        if preferred:
            preferred_docs = [d for d in candidates if d.metadata.get("category") == preferred]
            rest = [d for d in candidates if d.metadata.get("category") not in ("overview", preferred)]
            merged = overview + preferred_docs + rest
            return merged[:self.k]

        return candidates[:self.k]


# ─────────────────────────────────────────────────────────────────
# STEP 3d + 4  PromptTemplate + LLMChain
# ─────────────────────────────────────────────────────────────────

# Single-turn RAG prompt
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are InsightForge AI, an expert Business Intelligence analyst for a retail sales company.

Retrieved business data context:
{context}

Based strictly on the data above, answer the following question.
Always cite specific numbers ($, %, counts). Conclude with 1–2 actionable recommendations.

Question: {question}

Detailed Answer:"""
)

# Summary chain prompt (Step 4 — SequentialChain input)
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["raw_stats"],
    template="""You are a senior business analyst. Given these raw statistics:
{raw_stats}

Write a concise executive summary (3–4 sentences) highlighting the most important finding.
Executive Summary:"""
)

# Recommendation chain prompt (Step 4 — SequentialChain output)
RECOMMENDATION_PROMPT = PromptTemplate(
    input_variables=["executive_summary"],
    template="""Based on this business executive summary:
{executive_summary}

Provide exactly 3 specific, actionable strategic recommendations with measurable targets.
Strategic Recommendations:"""
)


def get_llm() -> ChatGroq:
    """Return configured ChatGroq LLM instance."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", ""),
    )


# ─────────────────────────────────────────────────────────────────
# STEP 4  Sequential chain for chained prompts
# ─────────────────────────────────────────────────────────────────
def build_sequential_chain() -> SequentialChain:
    """
    Step 4 — Chain prompts using LangChain SequentialChain.
    Chain 1: raw stats → executive summary
    Chain 2: executive summary → strategic recommendations
    """
    llm = get_llm()

    summary_chain = LLMChain(
        llm=llm,
        prompt=SUMMARY_PROMPT,
        output_key="executive_summary",
        verbose=False,
    )

    recommendation_chain = LLMChain(
        llm=llm,
        prompt=RECOMMENDATION_PROMPT,
        output_key="strategic_recommendations",
        verbose=False,
    )

    return SequentialChain(
        chains=[summary_chain, recommendation_chain],
        input_variables=["raw_stats"],
        output_variables=["executive_summary", "strategic_recommendations"],
        verbose=False,
    )


# ─────────────────────────────────────────────────────────────────
# STEP 5  FAISS vector store + RetrievalQA chain
# ─────────────────────────────────────────────────────────────────
def build_vectorstore(documents: List[Document]) -> FAISS:
    """
    Step 5 — Build FAISS vector store using HuggingFace embeddings.
    Uses all-MiniLM-L6-v2 (free, runs locally, no API key needed).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    split_docs = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


def build_rag_chain(vectorstore: FAISS) -> RetrievalQA:
    """
    Step 5 — RetrievalQA chain: custom retriever + ChatGroq + RAG_PROMPT.
    """
    llm = get_llm()
    retriever = InsightRetriever(vectorstore=vectorstore, k=5)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ─────────────────────────────────────────────────────────────────
# STEP 6  Conversational chain with ConversationBufferMemory
# ─────────────────────────────────────────────────────────────────
def build_conversational_chain(vectorstore: FAISS):
    """
    Step 6 — ConversationalRetrievalChain with ConversationBufferMemory.
    Retains full conversation context across turns.
    Returns (chain, memory) so memory can be inspected/cleared.
    """
    llm = get_llm()
    retriever = InsightRetriever(vectorstore=vectorstore, k=5)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": RAG_PROMPT} if False else {},
    )
    return chain, memory
