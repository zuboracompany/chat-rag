# -*- coding: utf-8 -*-

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# --- UI設定 ---
st.set_page_config(page_title="My RAG Chatbot (Pinecone)", layout="wide")
st.title("📄 ドキュメントQ&Aチャットボット (Pinecone版)")
st.markdown("""
このアプリは、アップロードしたPDFの内容について質問できるチャットボットです。
左のサイドバーからAPIキー等を設定し、PDFをアップロードして「学習を開始」ボタンを押してください。
""")
st.info("**注意:** Geminiの埋め込みモデル (`embedding-001`) は768次元です。PineconeでIndexを作成する際は、次元数(Dimensions)を`768`に設定してください。")


# --- APIキーと設定 ---
with st.sidebar:
    st.header("APIキー設定")
    # 環境変数から取得することを推奨
    google_api_key = st.text_input("Google API Key", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", value=os.environ.get("PINECONE_API_KEY", ""))
    pinecone_index_name = st.text_input("Pinecone Index Name", value=os.environ.get("PINECONE_INDEX_NAME", ""))

    st.header("ドキュメントのアップロード")
    uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type=['pdf'])

    # process_button
    process_button = st.button("学習を開始", disabled=not uploaded_file)


# --- メイン処理 ---
# APIキーが設定されていない場合は処理を中断
if not google_api_key or not pinecone_api_key or not pinecone_index_name:
    st.warning("左のサイドバーで全てのAPIキーとIndex名を設定してください。")
    st.stop()

# LangChainのモデルとVectorStoreを初期化
try:
    # Google Geminiのモデルを初期化
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", google_api_key=google_api_key, convert_system_message_to_human=True)
    
    # Pinecone VectorStoreを初期化
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
except Exception as e:
    st.error(f"モデルまたはVectorStoreの初期化中にエラーが発生しました: {e}")
    st.stop()


# --- ドキュメントの学習処理 ---
if process_button:
    if uploaded_file is not None:
        with st.spinner("ファイルを処理中... チャンキング -> ベクトル化 -> DB格納"):
            try:
                # 一時ファイルに保存して読み込む
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 1. ドキュメントの読み込み
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                # 2. テキストの分割 (チャンキング)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                docs = text_splitter.split_documents(documents)

                # 3. ベクトル化してPineconeに格納
                vectorstore.add_documents(docs)

                st.success("ドキュメントの学習が完了しました！")

                # 一時ファイルを削除
                os.remove(tmp_file_path)

            except Exception as e:
                st.error(f"ファイル処理中にエラーが発生しました: {e}")

# --- チャット処理 ---
# セッション状態でチャット履歴と会話チェーンを管理
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_chain" not in st.session_state:
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    st.session_state.conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザーからの入力を受け付ける
if prompt := st.chat_input("学習したドキュメントについて質問してください..."):
    # ユーザーのメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AIの回答を生成
    with st.spinner("回答を生成中..."):
        response = st.session_state.conversation_chain.invoke({"question": prompt})
        ai_response_text = response["answer"]

        # AIの回答を表示
        st.session_state.messages.append({"role": "assistant", "content": ai_response_text})
        with st.chat_message("assistant"):
            st.markdown(ai_response_text)

