import streamlit as st
import torch
import numpy as np
import nltk
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit.components.v1 as components
import warnings

warnings.filterwarnings('ignore')


######### КОНФИГУРАЦИЯ И ЗАГРУЗКА МОДЕЛЕЙ ###############

st.set_page_config(page_title="Детектор ИИ-текста", page_icon="🎓", layout="wide")

@st.cache_resource
def load_resources():
    """Загрузка NLTK пакетов и весов языковой модели"""
    for pkg in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)
        
   #model_id = "ai-forever/rugpt3medium_based_on_gpt2" 
    model_id = "fluently/FluentlyQwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval() 
    return tokenizer, model

tokenizer, model = load_resources()


#################  АНАЛИТИКА  ######################

def clean_text(text):
    """Исправляет склеенный при копировании текст, сохраняя структуру абзацев"""

    text = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', text)

    text = re.sub(r'([а-яёa-zА-ЯЁA-Z])([.?!:])([А-ЯЁA-Zа-яёa-z])', r'\1\2 \3', text)

    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def calculate_sentence_metrics(sentence):
    """Считает PPL и Top-10 для одного предложения"""
    if not sentence.strip(): return 100.0, 0.0
    
    encodings = tokenizer(sentence, return_tensors='pt')
    input_ids = encodings.input_ids
    if input_ids.size(1) < 2: return 100.0, 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        logits = outputs.logits

    ppl = float(torch.exp(loss).item())

    # Ранги
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    probs = torch.softmax(shift_logits, dim=-1)

    ranks = []
    for i in range(shift_labels.size(1)):
        true_token_id = shift_labels[0, i].item()
        token_probs = probs[0, i]
        sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
        rank = (sorted_indices == true_token_id).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = np.array(ranks)
    top10_ratio = np.sum(ranks < 10) / len(ranks) * 100 if len(ranks) > 0 else 0.0

    return ppl, top10_ratio


def analyze_text_pipeline(raw_text):
 
    #  Маскирование точек в списках до разбиения на предложения
  
    # Поиск: начало строки -> маркер списка или цифра -> короткий тезис -> точка -> Пробел + Заглавная буква
    plain_list_pat = re.compile(
        r'^([ \t]*(?:[•●▪◦■◆◇✓✔➢➤\-–—*#]|\d+[\.\)]?)\s+[А-Яа-яЁёA-Za-z0-9\s]{2,40})'
        r'\.'
        r'(\s+[А-ЯЁA-Z])',
        re.M
    )
    
    # Маскируем точку в списках
    masked_text = plain_list_pat.sub(r'\1_DOT_PLACEHOLDER_\2', raw_text)

    # Очищаем текст (склеенные слова, лишние пробелы)
    text = clean_text(masked_text)
    
  
    sentences = nltk.sent_tokenize(text, language="russian")
    
    analyzed_sentences = []
    total_ai_patterns = 0
    
    # Паттерн для поиска ИИ-списка 
    pat_list = re.compile(
        r'(?:^|\n)\s*(?:[•●▪◦■◆◇✓✔➢➤\-–—*#\d\.]+\s*)*' # Маркер списка
        r'([А-Яа-яЁёA-Za-z\s]{2,35})'                    # Текст тезиса (только буквы и пробелы)
        r'[:.]'                                         # Разделитель (двоеточие или точка)
        r'\s+[А-ЯЁ]'                                    # Пробел и Заглавная буква раскрытия
    )
    
    pat_dich1 = re.compile(r'(не просто|не только).{1,50}?но и', re.IGNORECASE)
    pat_dich2 = re.compile(r'не столько.+?сколько|не про.+?а про|не про.+?Это про', re.IGNORECASE)

    for sent in sentences:
        # Восстанавливаем точку для расчетов нейросети и для работы регулярки
        restored_sent = sent.replace("_DOT_PLACEHOLDER_", ".")
        
        ppl, top10 = calculate_sentence_metrics(restored_sent)
        words_count = len(restored_sent.split())
        print(f'local ppl {ppl}')
        warnings = []
        is_pattern = False
        
        if pat_list.search(restored_sent):
            warnings.append("Шаблон ИИ-списка (Тезис: Раскрытие)")
            is_pattern = True
        if pat_dich1.search(restored_sent):
            warnings.append("Дидактическая связка ('не только... но и')")
            is_pattern = True
        if pat_dich2.search(restored_sent):
            warnings.append("Паттерн противопоставления ('Это не про X, а про Y')")
            is_pattern = True
            
        if is_pattern:
            total_ai_patterns += 1
            
        analyzed_sentences.append({
            "text": restored_sent,
            "ppl": ppl,
            "top10": top10,
            "words_count": words_count,
            "is_pattern": is_pattern,
            "warnings": warnings
        })

    # Рассчитываем Burstiness по длинам предложений
    if len(sentences) > 1:
        burstiness = float(np.std([s["words_count"] for s in analyzed_sentences]))
    else:
        burstiness = 0.0
        
    words_all = [w.lower() for w in text.split() if w.isalpha()]
    ttr = float(len(set(words_all)) / len(words_all)) if words_all else 0.0
    
    # Для глобального расчета метрик документа также убираем плейсхолдеры
    restored_full_text = text.replace("_DOT_PLACEHOLDER_", ".")
    doc_ppl, doc_top10 = calculate_sentence_metrics(restored_full_text)
    print(doc_ppl)
    return {
        "doc_metrics": {
            "ppl": doc_ppl,
            "top10": doc_top10,
            "burstiness": burstiness,
            "ttr": ttr,
            "total_patterns": total_ai_patterns
        },
        "sentences": analyzed_sentences
    }


###############  ВЫЧИСЛЕНИЕ ВЕРОЯТНОСТИ И ПРОСМОТР ##################


def calculate_ai_probability(metrics):
    m = metrics
    score_ppl = max(0, min(100, (60 - m['ppl']) / (60 - 15) * 100))
    score_rank = max(0, min(100, (m['top10'] - 70) / (95 - 70) * 100))
    score_burst = max(0, min(100, (6.0 - m['burstiness']) / (6.0 - 1.5) * 100))
    score_ttr = max(0, min(100, (0.75 - m['ttr']) / (0.75 - 0.4) * 100))
    
    score_stylo = min(100.0, m['total_patterns'] * 50.0)
    
    final_prob = (0.35 * score_ppl) + (0.30 * score_rank) + (0.15 * score_burst) + (0.15 * score_stylo) + (0.05 * s_ttr if 's_ttr' in locals() else 0.05 * score_ttr)
    return round(final_prob, 1)

def generate_html_heatmap(sentences_data):
   
    html_content = """
    <style>
        .heatmap-container { font-family: Arial, sans-serif; line-height: 1.9; font-size: 16px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fff; }
        .text-sentence { border-radius: 3px; padding: 2px 4px; margin: 1px; cursor: pointer; transition: all 0.15s ease; }
        .text-sentence:hover { filter: brightness(0.9); box-shadow: 0px 2px 5px rgba(0,0,0,0.15); }
        .ai-pattern { border-bottom: 3px dashed #e67e22 !important; }
        
        /* Стили инспектора */
        #inspector-panel {
            margin-top: 20px;
            padding: 15px;
            background-color: #2c3e50;
            color: #fff;
            border-radius: 6px;
            font-size: 14px;
            border-left: 6px solid #3498db;
            transition: all 0.2s;
            min-height: 80px;
        }
    </style>
    
    <div class='heatmap-container'>
    """
    
    # Рендерим предложения
    for idx, s_data in enumerate(sentences_data):
        if s_data['ppl'] < 17.0:
            bg_color = "#ffb3b3"
            verdict = "ВЫСОКАЯ вероятность ИИ"
        elif s_data['ppl'] < 35.0:
            bg_color = "#ffe6b3"
            verdict = "СРЕДНЯЯ вероятность ИИ"
        else:
            bg_color = "#c6ecc6"
            verdict = "НИЗКАЯ вероятность ИИ"
            
        span_class = "text-sentence ai-pattern" if s_data['is_pattern'] else "text-sentence"
        

        clean_text_js = s_data['text'].replace("'", "\\'").replace('"', '\\"')
        warnings_js = ", ".join(s_data['warnings']).replace("'", "\\'")
        

        html_content += f"""
        <span class='{span_class}' 
              style='background-color: {bg_color};' 
              onmouseover="updateInspector('{verdict}', '{s_data['ppl']:.1f}', '{s_data['words_count']}', '{warnings_js}')"
              onmouseout="clearInspector()">
            {s_data['text']}
        </span> """
        

    html_content += """
    </div>
    
    <div id="inspector-panel">
        <b>Интерактивный инспектор:</b><br>
        <span style="color: #bdc3c7;">Наведите курсор на любое предложение на карте выше, чтобы увидеть подробный разбор параметров.</span>
    </div>

    <script>
        function updateInspector(verdict, ppl, words, warnings) {
            var panel = document.getElementById('inspector-panel');
            var color = "#3498db";
            if (verdict.includes("ВЫСОКАЯ")) color = "#e74c3c";
            else if (verdict.includes("СРЕДНЯЯ")) color = "#f39c12";
            else color = "#2ecc71";
            
            panel.style.borderLeftColor = color;
            
            var html = "<b>Результат анализа предложения:</b> <span style='color:" + color + "; font-weight:bold;'>" + verdict + "</span><br>";
            html += "• Локальная перплексия (PPL): <b>" + ppl + "</b> | • Длина: <b>" + words + "</b> слов<br>";
            if (warnings) {
                html += "<span style='color: #e67e22;'><b>⚠️ Обнаруженные паттерны ИИ:</b> " + warnings + "</span>";
            }
            panel.innerHTML = html;
        }
        
        function clearInspector() {
            var panel = document.getElementById('inspector-panel');
            panel.style.borderLeftColor = "#3498db";
            panel.innerHTML = "<b>Интерактивный инспектор:</b><br><span style='color: #bdc3c7;'>Наведите курсор на любое предложение на карте выше, чтобы увидеть подробный разбор параметров.</span>";
        }
    </script>
    """
    return html_content


############### STREAMLIT ИНТЕРФЕЙС ###################

st.title("Сертификат аутентичности текста")

user_input = st.text_area("Вставьте текст (минимум 3-4 предложения):", height=200)

if st.button("Провести анализ", type="primary"):
    if len(user_input.split()) < 15:
        st.warning("Текст слишком короткий для достоверного анализа.")
    else:
        with st.spinner("Анализ..."):
            

            analysis_results = analyze_text_pipeline(user_input)
            doc_metrics = analysis_results['doc_metrics']
            

            ai_prob = calculate_ai_probability(doc_metrics)
            

            heatmap_html = generate_html_heatmap(analysis_results['sentences'])
            
            # Вывод результатов
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Вероятность ИИ-генерации (P_ai)", value=f"{ai_prob}%")
                if ai_prob > 65: st.error("Скорее всего, текст сгенерирован средставми ИИ")
                elif ai_prob < 40: st.success("Скорее всего, текст написан человеком")
                else: st.warning("Вероятно, комбинация сгенерированного и оригинального текста")
                    
            with col2:
                st.markdown("### Общие маркеры:")
                if doc_metrics['total_patterns'] == 0:
                    st.write("✅ Структурных аномалий не обнаружено.")
                else:
                    st.warning(f"⚠️ Обнаружено {doc_metrics['total_patterns']} синтаксических паттернов ИИ. См. пунктир на карте.")

            st.divider()
            st.markdown("### Количественные параметры")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Перплексия (Норма >15)", round(doc_metrics['ppl'], 1))
            m2.metric("Слов из Top-10 (Норма < 75%)", f"{round(doc_metrics['top10'], 1)}%")
            m3.metric("Всплесковость (Норма > 4.0)", round(doc_metrics['burstiness'], 1))
            m4.metric("Лексическое разнообразие. (TTR)", round(doc_metrics['ttr'], 2))

            st.divider()
            st.markdown("### Цветовая карта предсказуемости")
            st.markdown("🟩 Человек | 🟨 Смешанно | 🟥 ИИ. \n *Оранжевый пунктир — обнаружен синтаксический паттерн ИИ.*")
            
  
            st.iframe(heatmap_html, height=450)
            
            # JSON Сертификат
            with st.expander("📄 Просмотреть  сертификат аутентичности (JSON)"):
                cert = {
                    "Ergonomic_Certificate": {
                        "Target": "Authenticity Assessment",
                        "Decision": "AI" if ai_prob > 65 else "HUMAN",
                        "Probability": ai_prob,
                        "Metrics": {
                            "PPL": round(doc_metrics['ppl'], 2),
                            "Rank_Top10_pct": round(doc_metrics['top10'], 2),
                            "Burstiness": round(doc_metrics['burstiness'], 2),
                            "TTR": round(doc_metrics['ttr'], 2)
                        },
                        "XAI_Tags": doc_metrics['total_patterns']
                    }
                }
                st.json(cert)