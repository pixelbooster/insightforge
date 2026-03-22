"""
InsightForge — AI-Powered Business Intelligence Assistant
Capstone Project | Advanced Generative AI

Part 2 (Step 7): Streamlit UI — LLMOps, Evaluation & Interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="InsightForge BI", page_icon="🔮", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');
  html,body,[class*="css"]{ font-family:'DM Mono',monospace; }
  .stApp{ background-color:#0a0c0f; }
  [data-testid="stSidebar"]{ background-color:#111418; border-right:1px solid #252c36; }
  #MainMenu,footer,header{ visibility:hidden; }
  .kpi-card{ background:#111418; border:1px solid #252c36; border-radius:10px; padding:18px 20px; }
  .kpi-card-amber{ border-top:2px solid #f5a623; }
  .kpi-card-green{ border-top:2px solid #34d399; }
  .kpi-card-blue { border-top:2px solid #60a5fa; }
  .kpi-card-purple{border-top:2px solid #a78bfa; }
  .kpi-label{ font-size:10px; color:#4a5568; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }
  .kpi-value{ font-family:'Syne',sans-serif; font-size:28px; font-weight:700; color:#e8eaed; line-height:1; margin-bottom:3px; }
  .kpi-sub  { font-size:11px; color:#8a95a3; }
  .sec-hdr  { font-family:'Syne',sans-serif; font-size:12px; font-weight:700; color:#8a95a3;
              text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;
              padding-bottom:6px; border-bottom:1px solid #252c36; }
  .chat-ai  { background:#181c22; border:1px solid #252c36; border-radius:10px;
              padding:12px 16px; margin:8px 0; margin-right:12%; color:#e8eaed; font-size:13px; line-height:1.7; }
  .chat-usr { background:rgba(245,166,35,0.08); border:1px solid rgba(245,166,35,0.2); border-radius:10px;
              padding:12px 16px; margin:8px 0; margin-left:12%; color:#e8eaed; font-size:13px; line-height:1.7; }
  .lbl-ai   { font-size:10px; color:#34d399; text-transform:uppercase; letter-spacing:1px; margin-bottom:3px; }
  .lbl-usr  { font-size:10px; color:#f5a623; text-transform:uppercase; letter-spacing:1px;
              margin-bottom:3px; text-align:right; }
  .stTabs [data-baseweb="tab-list"]{ background-color:#0a0c0f; border-bottom:1px solid #252c36; }
  .stTabs [data-baseweb="tab"]{ color:#4a5568; font-family:'DM Mono',monospace;
              font-size:12px; text-transform:uppercase; letter-spacing:1px; }
  .stTabs [aria-selected="true"]{ color:#f5a623 !important;
              border-bottom:2px solid #f5a623 !important; background:transparent !important; }
  .stTextInput input,.stTextArea textarea{ background:#181c22 !important; border:1px solid #252c36 !important;
              color:#e8eaed !important; font-family:'DM Mono',monospace !important; border-radius:8px !important; }
  .stButton>button{ background:#f5a623 !important; color:#0a0c0f !important; border:none !important;
              border-radius:8px !important; font-family:'DM Mono',monospace !important; font-weight:600 !important; }
  .stButton>button:hover{ background:#c47d10 !important; }
  .src-badge{ display:inline-block; background:rgba(96,165,250,0.1);
              border:1px solid rgba(96,165,250,0.25); border-radius:4px;
              padding:2px 8px; font-size:10px; color:#60a5fa; margin:2px 3px; }
  .mem-badge{ background:rgba(52,211,153,0.08); border:1px solid rgba(52,211,153,0.2);
              border-radius:6px; padding:6px 12px; font-size:11px; color:#34d399; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

COLORS = ["#f5a623","#34d399","#60a5fa","#a78bfa","#2dd4bf","#f87171"]
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="DM Mono", color="#8a95a3", size=11),
          margin=dict(l=10,r=10,t=36,b=10),
          xaxis=dict(gridcolor="#1e232b",linecolor="#252c36"),
          yaxis=dict(gridcolor="#1e232b",linecolor="#252c36"),
          colorway=COLORS)

# ── Data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data.csv", parse_dates=["Date"])
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Quarter"]   = df["Date"].dt.quarter
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["AgeGroup"]  = pd.cut(df["Customer_Age"], bins=[17,30,45,60,70],
                              labels=["18–30","31–45","46–60","61–70"])
    df["AgeGroup"]  = df["AgeGroup"].astype(str).replace("nan", "Unknown")
    return df

df = load_data()

# ── LangChain pipeline ────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Building FAISS vector store & LangChain pipeline…")
def build_pipeline():
    """
    Builds:
      • LangChain Documents from sales data (knowledge base)
      • FAISS vector store with HuggingFace embeddings
      • ConversationalRetrievalChain (Step 5 RAG + Step 6 Memory)
    """
    from rag_chain import build_documents, build_vectorstore, build_conversational_chain
    docs        = build_documents(df)
    vectorstore = build_vectorstore(docs)
    chain, memory = build_conversational_chain(vectorstore)
    return chain, memory, vectorstore

# ── Sidebar ───────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY","")

with st.sidebar:
    st.markdown('<div style="text-align:center;padding:16px 0 20px">'
                '<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e8eaed">'
                'Insight<span style="color:#f5a623">Forge</span></div>'
                '<div style="font-size:10px;color:#4a5568;letter-spacing:2px;margin-top:4px">BUSINESS INTELLIGENCE</div>'
                '</div>', unsafe_allow_html=True)
    st.markdown("---")

    if groq_key:
        st.markdown('<div style="font-size:11px;color:#34d399">● Groq API connected</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;color:#4a5568;margin-top:2px">LangChain · FAISS · llama-3.3-70b</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:11px;color:#f87171">● GROQ_API_KEY missing</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;color:#8a95a3;line-height:1.9">'
                f'📦 <b style="color:#e8eaed">{len(df):,}</b> transactions<br>'
                f'📅 <b style="color:#e8eaed">2022–2028</b><br>'
                f'🏷️ <b style="color:#e8eaed">4</b> products · <b style="color:#e8eaed">4</b> regions<br>'
                f'💰 <b style="color:#f5a623">${df["Sales"].sum()/1e6:.2f}M</b> revenue</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Filters</div>', unsafe_allow_html=True)
    sel_p = st.multiselect("Products", df["Product"].unique().tolist(), default=df["Product"].unique().tolist())
    sel_r = st.multiselect("Regions",  df["Region"].unique().tolist(),  default=df["Region"].unique().tolist())
    yr    = st.slider("Year range", int(df["Year"].min()), int(df["Year"].max()),
                      (int(df["Year"].min()), int(df["Year"].max())))
    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#4a5568;letter-spacing:1px">Capstone · Advanced Gen AI</div>', unsafe_allow_html=True)

fdf = df[df["Product"].isin(sel_p) & df["Region"].isin(sel_r) & df["Year"].between(yr[0],yr[1])]

# ── Header ────────────────────────────────────────────────────────
st.markdown('<div style="margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid #252c36">'
            '<div style="font-family:Syne,sans-serif;font-size:30px;font-weight:800;color:#e8eaed;line-height:1">'
            'Business <span style="color:#f5a623">Intelligence</span> Dashboard</div>'
            '<div style="font-size:11px;color:#4a5568;margin-top:5px;letter-spacing:0.5px">'
            'LangChain · FAISS RAG · ConversationBufferMemory · Groq llama-3.3-70b · QAEvalChain</div>'
            '</div>', unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi-card kpi-card-amber"><div class="kpi-label">💰 Total Revenue</div><div class="kpi-value">${fdf["Sales"].sum():,.0f}</div><div class="kpi-sub">Filtered dataset</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-card kpi-card-green"><div class="kpi-label">📈 Avg Sale</div><div class="kpi-value">${fdf["Sales"].mean():,.0f}</div><div class="kpi-sub">Median ${fdf["Sales"].median():,.0f}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi-card kpi-card-blue"><div class="kpi-label">⭐ Avg Satisfaction</div><div class="kpi-value">{fdf["Customer_Satisfaction"].mean():.2f}<span style="font-size:14px">/5</span></div><div class="kpi-sub">σ {fdf["Customer_Satisfaction"].std():.2f}</div></div>', unsafe_allow_html=True)
with k4:
    age_min = int(fdf["Customer_Age"].dropna().min()) if not fdf["Customer_Age"].dropna().empty else 0
    age_max = int(fdf["Customer_Age"].dropna().max()) if not fdf["Customer_Age"].dropna().empty else 0
    st.markdown(f'<div class="kpi-card kpi-card-purple"><div class="kpi-label">👥 Avg Age</div><div class="kpi-value">{fdf["Customer_Age"].mean():.1f}</div><div class="kpi-sub">Range {age_min}–{age_max} yrs</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4 = st.tabs(["📊  Charts","🎯  Segmentation","📋  Data Table","🤖  AI Assistant"])

with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Revenue by Product</div>', unsafe_allow_html=True)
        p_df = fdf.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales",ascending=False)
        fig = px.bar(p_df, x="Product", y="Sales", color="Product", color_discrete_sequence=COLORS, text_auto=".3s")
        fig.update_layout(**PL, showlegend=False, height=290); fig.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">Regional Distribution</div>', unsafe_allow_html=True)
        r_df = fdf.groupby("Region")["Sales"].sum().reset_index()
        fig = px.pie(r_df, values="Sales", names="Region", color_discrete_sequence=COLORS, hole=0.6)
        fig.update_layout(**PL, height=290); fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Annual Revenue Trend — Sales Performance by Time Period</div>', unsafe_allow_html=True)
    y_df = fdf.groupby("Year")["Sales"].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_df["Year"], y=y_df["Sales"], mode="lines+markers+text",
        line=dict(color="#f5a623",width=2.5), fill="tozeroy", fillcolor="rgba(245,166,35,0.07)",
        marker=dict(color="#f5a623",size=8,line=dict(color="#0a0c0f",width=2)),
        text=[f"${v/1000:.0f}K" for v in y_df["Sales"]], textposition="top center",
        textfont=dict(size=10,color="#8a95a3")))
    fig.update_layout(**PL, height=290); fig.update_xaxes(tickmode="array", tickvals=y_df["Year"].tolist())
    st.plotly_chart(fig, use_container_width=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sec-hdr">Monthly Seasonality</div>', unsafe_allow_html=True)
        m_df = fdf.groupby("Month")["Sales"].mean().reset_index()
        mlbl = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        bc = ["#f5a623" if v>565 else "#f87171" if v<540 else "#60a5fa" for v in m_df["Sales"]]
        fig = go.Figure(go.Bar(x=mlbl[:len(m_df)], y=m_df["Sales"], marker_color=bc, marker_line_width=0,
            text=[f"${v:.0f}" for v in m_df["Sales"]], textposition="outside", textfont_size=9))
        fig.update_layout(**PL, showlegend=False, height=290); st.plotly_chart(fig, use_container_width=True)
    with c4:
        st.markdown('<div class="sec-hdr">Customer Satisfaction by Product</div>', unsafe_allow_html=True)
        s_df = fdf.groupby("Product")["Customer_Satisfaction"].mean().round(3).reset_index().sort_values("Customer_Satisfaction")
        fig = px.bar(s_df, x="Customer_Satisfaction", y="Product", orientation="h",
                     color="Product", color_discrete_sequence=COLORS, text="Customer_Satisfaction")
        fig.update_layout(**PL, showlegend=False, height=290); fig.update_traces(marker_line_width=0, textfont_size=11)
        fig.update_xaxes(range=[2.8,3.2]); st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-hdr">Product × Region Revenue Heatmap</div>', unsafe_allow_html=True)
    heat = fdf.pivot_table(values="Sales", index="Product", columns="Region", aggfunc="sum")
    fig = go.Figure(go.Heatmap(z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
        colorscale=[[0,"#0a0c0f"],[0.5,"#c47d10"],[1,"#f5a623"]],
        text=[[f"${v:,.0f}" for v in row] for row in heat.values],
        texttemplate="%{text}", textfont_size=11))
    fig.update_layout(**PL, height=250); st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Revenue by Age Group</div>', unsafe_allow_html=True)
        ag = fdf.groupby("AgeGroup",observed=True)["Sales"].sum().reset_index()
        fig = px.bar(ag, x="AgeGroup", y="Sales", color="AgeGroup", color_discrete_sequence=COLORS, text_auto=".3s")
        fig.update_layout(**PL, showlegend=False, height=270); fig.update_traces(marker_line_width=0, textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="sec-hdr">Gender Split</div>', unsafe_allow_html=True)
        g_df = fdf.groupby("Customer_Gender")["Sales"].sum().reset_index()
        fig = px.pie(g_df, values="Sales", names="Customer_Gender", color_discrete_sequence=["#a78bfa","#60a5fa"], hole=0.55)
        fig.update_layout(**PL, height=260); fig.update_traces(textinfo="percent+label", textfont_size=12)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="sec-hdr">Sale Value vs Satisfaction</div>', unsafe_allow_html=True)
        bb = fdf.groupby("Product").agg(AvgSale=("Sales","mean"),AvgSat=("Customer_Satisfaction","mean"),Vol=("Sales","count")).reset_index()
        fig = px.scatter(bb, x="AvgSat", y="AvgSale", size="Vol", color="Product",
                         color_discrete_sequence=COLORS, text="Product")
        fig.update_layout(**PL, height=270); fig.update_traces(textposition="top center", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="sec-hdr">Sales Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x="Sales", nbins=30, color_discrete_sequence=["#f5a623"])
        fig.update_layout(**PL, showlegend=False, height=260); fig.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="sec-hdr">Revenue by Gender × Product</div>', unsafe_allow_html=True)
    gp = fdf.groupby(["Customer_Gender","Product"])["Sales"].sum().reset_index()
    fig = px.bar(gp, x="Product", y="Sales", color="Customer_Gender",
                 color_discrete_sequence=["#a78bfa","#60a5fa"], barmode="group", text_auto=".3s")
    fig.update_layout(**PL, height=290); fig.update_traces(marker_line_width=0, textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st1,st2,st3 = st.tabs(["Product summary","Regional summary","Raw data"])
    with st1:
        ps = fdf.groupby("Product").agg(Total_Revenue=("Sales","sum"),Avg_Sale=("Sales","mean"),
            Transactions=("Sales","count"),Avg_Satisfaction=("Customer_Satisfaction","mean"),
            Std_Dev=("Sales","std")).round(2).reset_index().sort_values("Total_Revenue",ascending=False)
        ps["Total_Revenue"] = ps["Total_Revenue"].map("${:,.0f}".format)
        ps["Avg_Sale"]      = ps["Avg_Sale"].map("${:,.1f}".format)
        ps["Std_Dev"]       = ps["Std_Dev"].map("${:,.1f}".format)
        st.dataframe(ps, use_container_width=True, hide_index=True)
    with st2:
        rs = fdf.groupby("Region").agg(Total_Revenue=("Sales","sum"),Avg_Sale=("Sales","mean"),
            Transactions=("Sales","count"),Avg_Satisfaction=("Customer_Satisfaction","mean")
            ).round(2).reset_index().sort_values("Total_Revenue",ascending=False)
        rs["Market_Share"] = (rs["Total_Revenue"]/rs["Total_Revenue"].sum()*100).round(1).astype(str)+"%"
        rs["Total_Revenue"] = rs["Total_Revenue"].map("${:,.0f}".format)
        rs["Avg_Sale"]      = rs["Avg_Sale"].map("${:,.1f}".format)
        st.dataframe(rs, use_container_width=True, hide_index=True)
    with st3:
        sc,so = st.columns([3,1])
        with sc: srch = st.text_input("🔍 Search", placeholder="e.g. Widget A or West")
        with so: srt  = st.selectbox("Sort by",["Date","Sales","Customer_Satisfaction","Customer_Age"])
        disp = fdf.copy()
        if srch:
            mask = (disp["Product"].str.contains(srch,case=False,na=False)|disp["Region"].str.contains(srch,case=False,na=False))
            disp = disp[mask]
        disp = disp.sort_values(srt,ascending=False)
        st.dataframe(disp[["Date","Product","Region","Sales","Customer_Age","Customer_Gender","Customer_Satisfaction"]].head(500),
                     use_container_width=True, hide_index=True)
        st.caption(f"Showing {min(500,len(disp)):,} of {len(disp):,} records")

with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    if not groq_key:
        st.warning("⚠️ Add GROQ_API_KEY to your .env file to enable the AI assistant.")
        st.code("GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx", language="bash")
        st.stop()

    ai_col,sug_col = st.columns([3,1])
    with sug_col:
        st.markdown('<div class="sec-hdr">💡 Quick prompts</div>', unsafe_allow_html=True)
        for icon,text in [
            ("📦","Which product generates the highest revenue?"),
            ("🌍","Compare regional performance — identify the weakest region."),
            ("📅","What are the seasonal sales trends across months?"),
            ("👥","Which customer age group drives the most revenue?"),
            ("⭐","Analyze customer satisfaction scores by product."),
            ("📈","What 3 strategic recommendations would you make?"),
            ("🎯","Top product-region combinations to prioritize?"),
            ("⚠️","What risks or anomalies exist in the data?"),
        ]:
            if st.button(f"{icon} {text}", key=f"s_{text[:12]}", use_container_width=True):
                st.session_state["pending"] = text

    with ai_col:
        st.markdown('<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;padding:12px 16px;'
                    'background:#111418;border:1px solid #252c36;border-radius:10px">'
                    '<div style="font-size:24px">🔮</div>'
                    '<div><div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;color:#e8eaed">InsightForge AI</div>'
                    '<div style="font-size:10px;color:#34d399;margin-top:2px">'
                    '● LangChain ConversationalRetrievalChain · FAISS · ConversationBufferMemory · Groq</div>'
                    '</div></div>', unsafe_allow_html=True)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        try:
            chain, memory, _ = build_pipeline()
            pipeline_ok = True
        except Exception as e:
            st.error(f"Pipeline init error: {e}")
            pipeline_ok = False
            chain = memory = None

        if memory:
            n = len(memory.chat_memory.messages)//2
            st.markdown(f'<div class="mem-badge">ConversationBufferMemory — {n} turn{"s" if n!=1 else ""} stored</div>',
                        unsafe_allow_html=True)

        # Chat display
        if not st.session_state.chat_history:
            st.markdown('<div class="lbl-ai">🔮 InsightForge AI</div>'
                        '<div class="chat-ai">Hello! I am your AI Business Intelligence assistant powered by '
                        '<b>LangChain ConversationalRetrievalChain</b>, <b>FAISS RAG</b>, and '
                        '<b>ConversationBufferMemory</b>.<br><br>'
                        'I have semantic access to your 2,500-transaction sales dataset (2022–2028) '
                        'and remember context across our conversation. What would you like to explore?</div>',
                        unsafe_allow_html=True)
        else:
            for role, msg, sources in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f'<div class="lbl-usr">You</div><div class="chat-usr">{msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="lbl-ai">🔮 InsightForge AI</div><div class="chat-ai">{msg}</div>',
                                unsafe_allow_html=True)
                    if sources:
                        badges = "".join(f'<span class="src-badge">{s}</span>' for s in sources)
                        st.markdown(f'<div style="margin-bottom:4px">{badges}</div>', unsafe_allow_html=True)

        ic,bc = st.columns([5,1])
        with ic:
            default = st.session_state.pop("pending","")
            user_input = st.text_input("msg", value=default,
                placeholder="Ask about sales, products, regions, demographics…",
                label_visibility="collapsed", key="chat_inp")
        with bc:
            send = st.button("Send ➤", use_container_width=True)

        if send and user_input.strip() and pipeline_ok and chain:
            q = user_input.strip()
            st.session_state.chat_history.append(("user", q, []))
            with st.spinner("🔍 Retrieving from FAISS · generating response…"):
                try:
                    result  = chain.invoke({"question": q})
                    answer  = result.get("answer", result.get("result",""))
                    srcdocs = result.get("source_documents",[])
                    sources = list({d.metadata.get("category","").replace("_"," ").title()
                                    for d in srcdocs if d.metadata.get("category")})
                    st.session_state.chat_history.append(("assistant", answer, sources))
                except Exception as e:
                    st.session_state.chat_history.append(("assistant", f"⚠️ Error: {e}", []))
            st.rerun()

        cc,cm = st.columns(2)
        with cc:
            if st.session_state.chat_history:
                if st.button("🗑️ Clear chat"):
                    st.session_state.chat_history = []
                    if memory: memory.clear()
                    st.rerun()
        with cm:
            if memory and st.button("🧠 Show memory buffer"):
                msgs = memory.chat_memory.messages
                if msgs:
                    with st.expander("ConversationBufferMemory", expanded=True):
                        for m in msgs:
                            role = "User" if m.type == "human" else "AI"
                            st.markdown(f"**{role}:** {m.content[:250]}{'…' if len(m.content)>250 else ''}")
                else:
                    st.info("Memory buffer is empty.")
