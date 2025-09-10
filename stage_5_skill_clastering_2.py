import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel
from IPython.display import display
# --- Инициализация модели и токенизатора ---
tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- пуллинг эмбеддингов ---
def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]


# --- функция минимального числа кластеров ---
def choose_min_clusters(n_skills, min_clusters=20):
    if n_skills < 60:
        n = math.ceil(n_skills * 0.95)
    elif n_skills < 100:
        n = math.ceil(n_skills * 0.4)
    elif n_skills < 200:
        n = math.ceil(n_skills * 0.25)
    elif n_skills < 400:    
        n = math.ceil(n_skills * 0.20)
    elif n_skills < 800:    
        n = math.ceil(n_skills * 0.18)
    elif n_skills < 1200:
        n = math.ceil(n_skills * 0.15)
    else:
        n = math.ceil(n_skills * 0.10)
    return max(min_clusters, n)

# --- функция получения эмбеддингов ---
def embed_skills(skills_list, model, tokenizer, pooling="cls", device="cpu", max_length=16):
    tokenized_inputs = tokenizer(
        skills_list, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    model = model.to(device)
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    embeddings = pool(outputs.last_hidden_state, tokenized_inputs["attention_mask"], pooling_method=pooling)
    return F.normalize(embeddings, p=2, dim=1).cpu().numpy()



# --- функция KMeans + merge ---
def kmeans_with_merge(embeddings, skills, n_clusters, sim_threshold=0.9, random_state=42):
    """
    KMeans + объединение похожих кластеров по косинусной близости центроидов
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # cluster -> список индексов
    cluster2idx = {}
    for i, cl in enumerate(labels):
        cluster2idx.setdefault(cl, []).append(i)

    # функция центроидов
    def compute_centroids(c2i):
        return {cl: np.mean(embeddings[idxs], axis=0) for cl, idxs in c2i.items()}
    
    cluster2centroid = compute_centroids(cluster2idx)

    # мердж похожих
    merged = True
    while merged:
        merged = False
        clusters = list(cluster2centroid.keys())
        if len(clusters) < 2:
            break
        
        sims = cosine_similarity([cluster2centroid[c] for c in clusters])
        np.fill_diagonal(sims, -1)
        
        i, j = divmod(sims.argmax(), sims.shape[0])
        if sims[i, j] >= sim_threshold:
            c1, c2 = clusters[i], clusters[j]
            cluster2idx[c1].extend(cluster2idx[c2])
            del cluster2idx[c2]
            cluster2centroid = compute_centroids(cluster2idx)
            merged = True

    # skill -> cluster
    skill2cluster = {}
    for new_cl, idxs in enumerate(cluster2idx.values()):
        for i in idxs:
            skill2cluster[skills[i]] = f"Cluster_{new_cl}"

    return skill2cluster


# --- главный пайплайн ---
def cluster_skills_by_position(df, model, tokenizer, skills_col="mapped_skills",
                               pooling="mean", device="cpu", 
                               min_clusters=20, sim_threshold=0.9):
    """
    Кластеризация скиллов по позиции с merge похожих кластеров.
    skills_col: колонка со списками скиллов (например, 'mapped_skills')
    
    Возвращает:
        - df_clustered: DataFrame с колонкой clustered_skills
        - position_skill2cluster: словарь skill -> кластер
        - cluster_examples: примеры топ-5 скиллов в каждом кластере
    """
    df = df.copy()
    df["clustered_skills"] = [[] for _ in range(len(df))]

    position_skill2cluster = {}
    cluster_examples = {}

    for position, group in df.groupby("position"):
        all_skills = sorted(set(s for skills in group[skills_col] for s in skills))
        n_skills = len(all_skills)

        if n_skills == 0:
            continue

        n_clusters = choose_min_clusters(n_skills, min_clusters=min_clusters)

        if n_skills < n_clusters:
            skill2cluster = {s: f"Cluster_{i}" for i, s in enumerate(all_skills)}
            clusters_after_merge = len(skill2cluster)
        else:
            embeddings = embed_skills(all_skills, model, tokenizer, pooling=pooling, device=device)
            skill2cluster = kmeans_with_merge(
                embeddings, all_skills,
                n_clusters=n_clusters,
                sim_threshold=sim_threshold
            )
            clusters_after_merge = len(set(skill2cluster.values()))

        # добавляем префикс позиции
        skill2cluster = {s: f"{position}__{cl}" for s, cl in skill2cluster.items()}
        position_skill2cluster.update(skill2cluster)

        # обновляем DataFrame
        df.loc[group.index, "clustered_skills"] = group[skills_col].apply(
            lambda x: sorted({skill2cluster.get(s, s) for s in x}) if isinstance(x, list) else []
        )

        # сохраняем примеры топ-5 скиллов из каждого кластера
        cluster2skills = defaultdict(list)
        for skill, cl in skill2cluster.items():
            cluster2skills[cl].append(skill)

        cluster_examples[position] = {cl: sorted(s)[:5] for cl, s in cluster2skills.items()}

        # логирование
        print(f"▶ Позиция: {position}")
        print(f"   Уникальных скиллов: {n_skills}")
        print(f"   Минимальное число кластеров: {n_clusters}")
        print(f"   Кластеров после merge: {clusters_after_merge}\n")

    return df, position_skill2cluster, cluster_examples

df_train_result= pd.read_csv('data/stage_5_df_train_result_vacancy_clastered_l1_v9.csv')
import ast
import json
from collections import Counter

# -------------------------------
# 1) Функция надёжной нормализации одной ячейки mapped_skills
# -------------------------------
def normalize_skills_cell(x):
    """
    Приводит значения разных форматов к list[str].
    Обрабатывает: None/NaN, list, tuple, set, строковые представления Python-списков или JSON, 
    строки с разделителем запятой, одиночные строки.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [str(i).strip() for i in list(x) if str(i).strip()]
    if pd.isna(x):
        return []
    # строка
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        # 1) попробовать ast.literal_eval (строки вида "['Python','SQL']" или "('a','b')")
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return [str(i).strip() for i in list(val) if str(i).strip()]
        except Exception:
            pass
        # 2) пробуем JSON (["a","b"])
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return [str(i).strip() for i in val if str(i).strip()]
        except Exception:
            pass
        # 3) если есть запятые — разложим по запятой
        if ',' in s:
            return [item.strip() for item in s.split(',') if item.strip()]
        # 4) иначе считаем как один скилл
        return [s]
    # остальные типы — приводим к строке
    return [str(x).strip()] if str(x).strip() else []

# -------------------------------
# 2) Функция дебага: сколько скиллов определяется "на старте"
# -------------------------------
def debug_skill_loading(df, position_col="position", skills_col="mapped_skills", top_n_positions=30, sample_n=5):
    """
    Возвращает таблицу со статистикой по позициям и печатает образцы исходных и нормализованных ячеек.
    """
    df_local = df.copy()
    # НЕ ЗАМЕНЯЕМ оригинал — создаём временную колонку
    df_local["_norm_skills"] = df_local[skills_col].apply(normalize_skills_cell)
    rows = []
    for pos, group in df_local.groupby(position_col):
        total_rows = len(group)
        # типы оригинальных ячеек
        type_counts = group[skills_col].apply(lambda x: type(x).__name__).value_counts().to_dict()
        # сколько пустых после нормализации
        empty_after = (group["_norm_skills"].apply(len) == 0).sum()
        # уникальные скиллы после нормализации
        uniq = set()
        for lst in group["_norm_skills"]:
            uniq.update(lst)
        rows.append({
            "Позиция": pos,
            "Строк (всего)": total_rows,
            "Типы оригинала (top)": dict(list(type_counts.items())[:3]),
            "Пустых после нормализации": empty_after,
            "Пустых %": round(100 * empty_after / max(1, total_rows), 1),
            "Уникальных скиллов (после норм.)": len(uniq)
        })
    stats_df = pd.DataFrame(rows).sort_values("Уникальных скиллов (после норм.)", ascending=False).reset_index(drop=True)

    # печатаем топ problematic позиций (по количеству пустых или нулей)
    print("\n=== Статистика по позициям (топ) ===")
    display(stats_df.head(top_n_positions))

    # Покажем примеры для позиций, где 'Уникальных' == 0 или где много пустых
    problematic = stats_df[(stats_df["Уникальных скиллов (после норм.)"]==0) | (stats_df["Пустых %"] > 50)]
    if problematic.shape[0] == 0:
        problematic = stats_df.head(5)
    print("\n=== Примеры исходных и нормализованных ячеек для проблемных позиций ===")
    for pos in problematic["Позиция"].tolist()[:10]:
        print("\n--- Позиция:", pos)
        sample = df_local[df_local[position_col]==pos].head(sample_n)
        for idx, row in sample.iterrows():
            print(f"index={idx}")
            print("  raw  ->", repr(row[skills_col]))
            print("  norm ->", row["_norm_skills"])
    return stats_df, df_local

# -------------------------------
# 3) Быстрая функция для массовой коррекции (если нужно)
# -------------------------------
def force_normalize_column(df, skills_col="mapped_skills"):
    """
    Приводит колонку mapped_skills к типу list[str] (перезаписывает колонку).
    Рекомендуется запускать ТОЛЬКО после того, как вы проверили вывод debug_skill_loading.
    """
    df = df.copy()
    df[skills_col] = df[skills_col].apply(normalize_skills_cell)
    return df
# Нормализуем mapped_skills

df_train_result = force_normalize_column(df_train_result, "mapped_skills")
# --- Вызываем функцию кластеризации ---
df_clustered, position_skill2cluster, cluster_examples = cluster_skills_by_position(
    df=df_train_result,
    model=model,
    tokenizer=tokenizer,
    skills_col="mapped_skills",  
    pooling="mean",
    device=device,
    min_clusters=20,
    sim_threshold=0.8
)
def skills_cluster_stats(df, position_col="position", skills_col="mapped_skills", 
                         clustered_col="clustered_skills", position_skill2cluster=None, top_examples=5):
    """
    Возвращает статистику по каждой позиции:
    - количество уникальных скиллов (до кластеризации)
    - количество кластеров (после)
    - процент сжатия
    - примеры объединений (по 5 скиллов на кластер)
    
    position_skill2cluster: словарь skill -> кластер (нужен для построения примеров объединений)
    """
    stats = []

    for position, group in df.groupby(position_col):
        # --- уникальные скиллы до кластеризации ---
        all_orig_skills = set(s for skills in group[skills_col] for s in skills)

        # --- формируем кластер -> скиллы ---
        cluster2skills = defaultdict(set)
        for orig_skills, clust_skills in zip(group[skills_col], group[clustered_col]):
            if isinstance(orig_skills, list) and isinstance(clust_skills, list):
                for cl in clust_skills:
                    # добавляем оригинальные скиллы в кластер
                    cluster2skills[cl].update(orig_skills)

        # --- примеры объединений (топ по размеру кластера) ---
        examples = []
        sorted_clusters = sorted(cluster2skills.items(), key=lambda x: -len(x[1]))
        for cl, skills_set in sorted_clusters[:top_examples]:
            examples.append(f"{cl}: {', '.join(list(skills_set)[:6])}")  # 5-6 топ скиллов

        # --- пересчёт сжатия ---
        n_clusters = len(cluster2skills)
        compression = round(100 * (1 - n_clusters / max(1, len(all_orig_skills))), 1)

        stats.append({
            "Позиция": position,
            "Уникальных скиллов (было)": len(all_orig_skills),
            "Кластеров (стало)": n_clusters,
            "Сжатие (%)": compression,
            "Примеры объединений": examples
        })

    # не ограничиваем ширину столбца в pandas
    pd.set_option('display.max_colwidth', None)
    return pd.DataFrame(stats).sort_values("Сжатие (%)", ascending=False).reset_index(drop=True)

stats_df = skills_cluster_stats(
    df_clustered, 
    position_col="position", 
    skills_col="mapped_skills", 
    clustered_col="clustered_skills",
    position_skill2cluster=position_skill2cluster,
    top_examples=5
)

display(stats_df)
# сохраняем маппинг на диск
import joblib
joblib.dump(position_skill2cluster, "data/skill2cluster_mapping_9.pkl")
df_clustered.to_csv('data/stage_5_df_train_result_skills_clastered_l1_v9', index= False)
