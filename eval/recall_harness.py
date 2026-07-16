"""Stage1/Stage2 recall検証ハーネス（プレースホルダー）。

現時点ではスコープ外（骨組み構築フェーズ、検証は後続でユーザーが実施する）。
将来ここに実装する内容（federated-tickling-turtle.mdのLayer1設計を踏襲）:
  - GTのvisit区間内に、フラグ済みフレーム/ビン代表候補が最低1つでも重なるかの二値ヒット判定
  - MOG2併用 vs YOLO単独のhead-to-head比較（処理時間・候補削減率・ビン単位Recall・Precision）
  - Go/No-goライン（recall 0.97-0.99）との突き合わせ
"""

raise NotImplementedError(
    "recall_harness.py はプレースホルダーです。骨組み構築フェーズの完了後、"
    "ユーザー主導の検証フェーズで実装します。"
)
