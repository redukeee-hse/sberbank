import json
from collections import Counter, defaultdict


def load_sessions(path="sessions.jsonl"):
   sessions = []
   with open("sessions.jsonl") as f:
      for line in f:
         line = line.strip()
         if line:
               sessions.append(json.loads(line))

   # sessions — список списков целых чисел
   print(f"Всего сессий: {len(sessions)}")
   print(f"Первая сессия: {sessions[0]}")
   return sessions


def train_test_split(
   sessions: list[list[int]],
   ) -> tuple[list[list[int]], list[int]]:

   train_sessions = [s[:-1] for s in sessions]
   test_targets = [s[-1] for s in sessions]
   # print(f"train_sessions: {train_sessions}", train_sessions)
   # print(f"test_targets: {test_targets}", test_targets)
   return train_sessions, test_targets


def hit_at_k(
   recommendations: list[list[int]],
   true_items: list[int],
   k: int = 10,
   ) -> float:
   hits = 0
   for recs, true_item in zip(recommendations, true_items):
      if true_item in recs[:k]:
         hits += 1
   return hits / len(true_items)


class TransitionRecommender:
   def __init__(self):
      self.transition_counts = defaultdict(Counter)
      self.transition_probs = {}
      self.item_counts = Counter()
      self.global_popular = []

   def fit(self, train_sessions):
      for session in train_sessions:
         self.item_counts.update(session)
         for a, b in zip(session, session[1:]):
               self.transition_counts[a][b] += 1

      self.global_popular = [item for item, _ in self.item_counts.most_common()]

      for item_i, next_counter in self.transition_counts.items():
         total = sum(next_counter.values())
         sorted_items = sorted(
            next_counter.items(),
            key=lambda x: (-x[1], x[0])
         )

         probs_for_item = {}
         for j, c in sorted_items:
            probs_for_item[j] = c / total

         self.transition_probs[item_i] = probs_for_item
      return self

   def recommend(self, session_history, k=10):
      last_item = session_history[-1]
      recs = list(self.transition_probs.get(last_item, {}).keys())
      seen = set(session_history)
      recs = [x for x in recs if x not in seen]

      if len(recs) < k:
         for item in self.global_popular:
               if item not in seen and item not in recs:
                  recs.append(item)
               if len(recs) == k:
                  break
      return recs[:k]

   def recommend_batch(self, train_sessions, k=10):
      return [self.recommend(session, k=k) for session in train_sessions]


if __name__ == "__main__":
   sessions = load_sessions("sessions.jsonl")
   train_sessions, test_targets = train_test_split(sessions)

   model = TransitionRecommender().fit(train_sessions)
   model_recommendations = model.recommend_batch(train_sessions, k=10)
   model_hit10 = hit_at_k(model_recommendations, test_targets, k=10)

   baseline_top10 = model.global_popular[:10]
   baseline_recs = [baseline_top10[:] for _ in range(len(test_targets))]
   baseline_hit10 = hit_at_k(baseline_recs, test_targets, k=10)

   print("Hit@10 модели:", round(model_hit10, 6))
   print("Hit@10 baseline:", round(baseline_hit10, 6))
   # print("Топ 10 популярных товаров:", baseline_top10)
