# Evaluation results

- Embedding model: `BAAI/bge-small-en-v1.5`
- Top-k: 5

## 1. Retrieval - chunking comparison (dense)

| Chunking | Recall@5 | Recall@1 | MRR | Hits / Total |
| --- | --- | --- | --- | --- |
| paragraph | 91.67% | 83.33% | 0.854 | 11/12 |
| article | 91.67% | 83.33% | 0.875 | 11/12 |

## 2. Retrieval - dense vs hybrid (bonus, paragraph chunking)

| Retrieval | Recall@5 | Recall@1 | MRR | Hits / Total |
| --- | --- | --- | --- | --- |
| dense | 91.67% | 83.33% | 0.854 | 11/12 |
| hybrid | 75.00% | 66.67% | 0.694 | 9/12 |

## 3. Answer quality

| Metric | Score | Count |
| --- | --- | --- |
| Correct citation (in-scope) | 91.67% | 11/12 |
| Correct 'Not found' (out-of-scope) | 100.00% | 3/3 |

## 4. Error analysis

Retrieval misses (paragraph chunking):
- Q1 (gold [5]): retrieved articles [17, 7, 10, 17, 17]

Answer failures:
- Q1 (gold Article [5]): cited [17] -> 'Not simply forever: you can request erasure if your CV is no longer necessary for the purp'
