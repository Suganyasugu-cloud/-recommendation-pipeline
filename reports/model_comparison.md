# Model Comparison (K=10)

| model      |   precision@10 |   recall@10 |   ndcg@10 |   coverage |   diversity@10 | rmse   | mae    |   n_eval_users |
|:-----------|---------------:|------------:|----------:|-----------:|---------------:|:-------|:-------|---------------:|
| popularity |         0.0768 |      0.0743 |    0.0992 |     0.0465 |         0.6941 | 1.0703 | 0.8571 |            905 |
| item_knn   |         0.1207 |      0.1398 |    0.1697 |     0.2784 |         0.7006 | n/a    | n/a    |            905 |
| svd_mf     |         0.0575 |      0.0516 |    0.0707 |     0.1209 |         0.7581 | 0.9876 | 0.7841 |            905 |

RMSE/MAE: popularity = item mean rating; svd_mf = bias + latent factors; item_knn is ranking-only (NaN). Users evaluated: those with at least one test rating >= 4 who appear in train. Seen items are excluded.
