import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import ALS

# ===== 1. ПОДГОТОВКА ДАННЫХ =====
# Пример: пользователи смотрят фильмы
data = pd.DataFrame({
    'user_id': [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3],
    'item_id': [0, 1, 2, 1, 3, 4, 0, 2, 1, 2, 4],
    'rating':  [5, 3, 4, 4, 5, 2, 5, 3, 4, 5, 3]
})

n_users = data['user_id'].max() + 1
n_items = data['item_id'].max() + 1

# Создаем UI матрицу (user × item)
user_item = sp.csr_matrix(
    (data['rating'], (data['user_id'], data['item_id'])),
    shape=(n_users, n_items)
)

# Транспонируем для fit() - получаем IU матрицу (item × user)
item_user = user_item.T.tocsr()

print(f"Матрица user_item: {user_item.shape}")  # (4, 5)
print(f"Матрица item_user: {item_user.shape}")  # (5, 4)

# ===== 2. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ =====
model = ALS(
    factors=32,              # Размерность латентных факторов
    regularization=0.01,     # Регуляризация
    iterations=20,           # Число итераций
    calculate_training_loss=True,
    random_state=42
)

# Обучение на IU матрице
model.fit(item_user, show_progress=True)

# ===== 3. РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ =====
user_id = 0

# Получить топ-3 рекомендации
recommendations = model.recommend(
    userid=user_id,
    user_items=user_item[user_id],  # UI формат!
    N=3,
    filter_already_liked_items=True
)

# Результат: (item_ids, scores)
items, scores = recommendations
print(f"\n📺 Рекомендации для пользователя {user_id}:")
for item, score in zip(items, scores):
    print(f"  Фильм {item}: score = {score:.4f}")

# ===== 4. ПОХОЖИЕ ТОВАРЫ =====
item_id = 1

similar = model.similar_items(itemid=item_id, N=3)
items_sim, scores_sim = similar

print(f"\n🎬 Фильмы похожие на {item_id}:")
for item, score in zip(items_sim, scores_sim):
    print(f"  Фильм {item}: similarity = {score:.4f}")

# ===== 5. ПОХОЖИЕ ПОЛЬЗОВАТЕЛИ =====
similar_users = model.similar_users(userid=user_id, N=2)
users_sim, scores_sim = similar_users

print(f"\n👥 Пользователи похожие на {user_id}:")
for user, score in zip(users_sim, scores_sim):
    print(f"  User {user}: similarity = {score:.4f}")

# ===== 6. ОБЪЯСНЕНИЕ РЕКОМЕНДАЦИИ =====
# Почему рекомендуем item=3 пользователю user=0?
explanation = model.explain(
    userid=user_id,
    user_items=user_item[user_id],
    itemid=3,
    N=2
)
items_exp, scores_exp = explanation

print(f"\n💡 Почему рекомендуем фильм 3 пользователю {user_id}:")
for item, score in zip(items_exp, scores_exp):
    print(f"  Потому что смотрел фильм {item}: вклад = {score:.4f}")

# ===== 7. ПРЕДСКАЗАНИЕ РЕЙТИНГА =====
# Скор для конкретной пары user-item
user_vector = model.user_factors[0]
item_vector = model.item_factors[3]
predicted_score = np.dot(user_vector, item_vector)

print(f"\n⭐ Предсказанный рейтинг (user 0, item 3): {predicted_score:.4f}")

# ===== 8. РАБОТА С ФАКТОРАМИ =====
print(f"\n📊 Размерности:")
print(f"  User факторы: {model.user_factors.shape}")  # (4, 32)
print(f"  Item факторы: {model.item_factors.shape}")  # (5, 32)

# Эмбеддинг пользователя 0
print(f"\n  User 0 эмбеддинг (первые 5): {user_vector[:5]}")

# ===== 9. БАТЧ РЕКОМЕНДАЦИИ ДЛЯ ВСЕХ ПОЛЬЗОВАТЕЛЕЙ =====
all_recommendations = model.recommend(
    userid=np.arange(n_users),  # Все пользователи
    user_items=user_item,       # Вся матрица
    N=3
)

print(f"\n🎯 Топ-3 для каждого пользователя:")
for uid, (items, scores) in enumerate(zip(*all_recommendations)):
    print(f"  User {uid}: items {items} (scores: {scores})")

# ===== 10. ХОЛОДНЫЙ СТАРТ - НОВЫЙ ПОЛЬЗОВАТЕЛЬ =====
# Новый пользователь посмотрел фильмы 0 и 2
new_user_interactions = sp.csr_matrix(
    ([4, 5], ([0, 0], [0, 2])),  # рейтинги 4 и 5
    shape=(1, n_items)
)

new_user_recs = model.recommend(
    userid=0,  # Любой ID (не используется)
    user_items=new_user_interactions[0],
    N=3,
    recalculate_user=True  # ВАЖНО! Пересчитываем факторы
)

print(f"\n🆕 Рекомендации для нового пользователя:")
for item, score in zip(*new_user_recs):
    print(f"  Фильм {item}: score = {score:.4f}")

# ===== 11. ОЦЕНКА КАЧЕСТВА =====
from implicit.evaluation import train_test_split, precision_at_k, ndcg_at_k

# Разбиваем данные
train, test = train_test_split(user_item, train_percentage=0.8)

# Обучаем на train
train_item_user = train.T.tocsr()
model_eval = ALS(factors=32, iterations=15, random_state=42)
model_eval.fit(train_item_user, show_progress=False)

# Метрики
p_at_5 = precision_at_k(model_eval, train, test, K=5)
ndcg = ndcg_at_k(model_eval, train, test, K=5)

print(f"\n📈 Метрики качества:")
print(f"  Precision@5: {p_at_5:.4f}")
print(f"  NDCG@5: {ndcg:.4f}")

# ===== 12. СОХРАНЕНИЕ/ЗАГРУЗКА =====
import pickle

# Сохранение
with open('als_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Загрузка
with open('als_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Проверка
test_recs = loaded_model.recommend(0, user_item[0], N=3)
print(f"\n💾 Модель загружена, рекомендации: {test_recs[0]}")
