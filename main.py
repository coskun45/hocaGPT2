from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
import gradio as gr
import re
import json
import chromadb
from datetime import datetime
load_dotenv()
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

def get_chroma_client(path):
    return  chromadb.PersistentClient(path)
    

def get_context(query: str, client, k) ->list:
    collection_list = client.list_collections()
    ergebnisse=[]
    query_vek = embedding_function.embed_query(query)
    for col_name in collection_list:
        name=col_name.name
        col = client.get_collection(name=name)
        res = col.query(query_vek,n_results=3)
        ergebnisse.append(res)
        extract_infos = extract_info_from_vektordb(ergebnisse)
    return extract_infos[:k]

def extract_info_from_vektordb(ergebnisse):
    result = []
    for e in ergebnisse:
        distances = e['distances'][0]
        
        indices = [idx for idx, value in enumerate(distances) if value < 1]
        if len(indices)>0:
            for idx in indices:
                item = { }
                
                item["ids"] = e['ids'][0][idx]
                item["distances"] = e['distances'][0][idx]
                item["metadatas"] = e['metadatas'][0][idx]
                item["documents"] = e['documents'][0][idx]
                
                
                result.append(item)
    return sorted(result, key=lambda x: x["distances"])
    

def generate_response(query: str, retrieved_docs):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0) 
    prompt = hub.pull("rlm/rag-prompt")
    result = {"query": query, "retrieved_docs": retrieved_docs, "response": None}
    generation_chain = prompt | llm | StrOutputParser()
    result["response"] = generation_chain.invoke({"context": retrieved_docs, "question": query})
    return result
        

def is_meaningful_input(user_input: str) -> bool:
    """
    Kullanıcıdan gelen inputun anlamlı olup olmadığını kontrol eder.
    
    Args:
        user_input (str): Kullanıcıdan gelen metin.
    
    Returns:
        bool: Anlamlı ise True, değilse False.
    """
    # Boş veya sadece boşluklardan oluşuyorsa False döndür
    if not user_input.strip():
        return False
    
    # Sadece semboller veya rastgele karakterler varsa False döndür
    # Alfanümerik (harf veya rakam) karakterlerin olup olmadığını kontrol eder
    if not re.search(r'[a-zA-Z0-9]', user_input):
        return False
    
    # Özel bir uzunluk kontrolü (ör. en az 3 karakter anlamlıdır)
    if len(user_input.strip()) < 3:
        return False
    
    return True

def create_context(result):
    contexts =""
    for r in result:
        contexts +=r['documents']
    return contexts

def create_source_infos(retrieved_docs):
    source_infos = {
    "page":"",
    "source":""
}
    source_list = []
    for r in retrieved_docs:
        source_infos['source']= (r["metadatas"]['source'].split('/')[-1])
        source_infos['page'] = r["metadatas"]["page"]
        source_list.append(source_infos.copy())
    return source_list

def format_response(answer, sources):
    formatted_sources = "\n".join(
        [f"- **Sayfa {src['page']}**: {src['source']}" for src in sources]
    )
    return f"""
### Cevap:
{answer}

### Kaynaklar:
{formatted_sources}

## ❗ Önemli Uyari ❗:
### HocaGPT hata yapabilir. Bu sebeble HocaGPT nin fetvalari ile amel etmeden önce dogrulugunu kontrol ediniz. 
"""
import json
# Document türündeki verileri JSON'a dönüştür
def document_to_dict(documents):
    dict_list = []
    for doc in documents:
        doc_dict ={ "id": None, "metadata": None, "page_content": None,"relevance_score": None}
        
        doc_dict['id']=doc['ids']
        doc_dict['metadata']=doc['metadatas']
        doc_dict['page_content']=doc['documents']
        doc_dict['relevance_score']=doc['distances']
        dict_list.append(doc_dict)
    return dict_list
counter = 1
def save_chat_history(retrieved_docs, gpt_response, message):
    from datetime import datetime
    history = {"id":None,"retrieved_docs": None, "gpt_response": None, "message": None, "zeitstempel": None}
    retrieved_docs = document_to_dict(retrieved_docs)
    history['retrieved_docs']= retrieved_docs
    history['response']=gpt_response
    history['message']=message
    history['zeitstempel']=datetime.now().isoformat()
    global counter
    history['id']=counter
    counter +=1
    import json

# 1. JSON dosyasını oku
    with open("chat_history.json", "r", encoding="utf-8") as dosya:
        veri = json.load(dosya)
    veri.append(history)
    
    with open("chat_history.json", "w") as json_file:
            json.dump(veri, json_file, indent=4)

    
    
def predict(message, history):
    history_langchain_format = []
    history ={}
    print(history_langchain_format)
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    print(history_langchain_format)
    if is_meaningful_input(message):
        history_langchain_format.append(HumanMessage(content=message))
        retrieved_docs = get_context(message, get_chroma_client("vectorDB"),k=3)
        if retrieved_docs[0]['distances']>1:
            return "Sormus oldugunuz soru HocaGPT nin kapsami disinda oldugundan cevap veremiyorum. Lütfen konu ile ilgili bir soru sormayi deneyin."
        else:
            context = create_context(retrieved_docs)
            gpt_response = generate_response(message, context)
            source_infos = create_source_infos(retrieved_docs)
            history_langchain_format.append(AIMessage(content=gpt_response['response']))
            response = format_response(gpt_response['response'], source_infos)
            save_chat_history(retrieved_docs, gpt_response['response'], message)
            print("SORU: ", message)
            print("************************************")
            print("CONTEXT: ", context)
            print("************************************")
            print("RESPONSE: ", gpt_response['response'])
            return response
    else:
        return "Lütfen anlamlı bir soru sorunuz."
demo = gr.ChatInterface(
    predict,
    type="messages",
    title="💬 HocaGPT 💬",
    save_history=True
    
)

demo.launch(auth_message="""
    <div style="text-align: center; font-family: Arial, sans-serif;">
        <h1 style="color: #2C3E50; font-size: 32px;">HocaGPT'ye Hoş Geldiniz</h1>
        <p style="font-size: 16px; color: #34495E;">
            <strong style="color: red;">❗ Chat esnasında girmiş olduğunuz bilgiler, uygulamanın iyileştirilmesi amacıyla kayıt altına alınacaktır. ❗</strong>
        </p>
        <p style="font-size: 14px; color: #7F8C8D;">
            Bunu kabul ediyorsanız lütfen kullanıcı adı ve şifrenizi giriniz.
        </p>
        <p>🚀 A chatbot powered by Coskun</>
    </div>
    """, auth=("admin","pass1234"), server_port=10000)