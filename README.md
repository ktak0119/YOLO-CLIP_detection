# YOLO-CLIP_detection

> **開発中（骨組み構築フェーズ）**: このリポジトリは実運用グレードの精度検証を終えたパイプラインでは
> ありません。まず6ステージが一通りデータで動く骨組みを作ることを目標にしており、recall/precisionの
> 検証・閾値の本決定・対象間のパラメータ共有可否の判断はこれから行います。

固定カメラで撮影した長時間動画から、昆虫（マルハナバチ・蝶など多様な対象）の訪花イベント候補を
検出するための汎用パイプライン。画像処理は5秒固定ビン単位で行い、訪花回数への変換（イベント統合）は
別工程として扱う。

## パイプライン構成

```
Stage1  MOG2動体検知（全フレーム連続実行、フラグのみ・クリップ非生成。任意処理、config切替可）
Stage2  YOLO検知＋ビンごとの代表クリップ抽出（conf>=0.30、1フレーム最大1クリップ、ビン内スコア上位K枚）
Stage3  CLIPスコアリング（ビン代表画像、足切りなし・スコア付与のみ）
Stage4  YOLOスコア＋CLIPスコアの統合評価・抽出（YOLO単独/CLIP単独/単純加算/ロジスティック回帰の比較）
Stage5  人間目視フィルタリング
Stage6  ビン結合（同時間帯・隣接ビンの機械的マージ）→クリップ動画出力
```

各ステージ間の受け渡しはJSON/CSV。設定は`configs/pipeline.yaml`（全対象共通の既定値）と
`configs/targets/*.yaml`（対象ごとに変えてよい項目のみ）に分離している。

対象ごとに変えてよいのは次の3点のみ（それ以外の数値パラメータは対象非依存の共通値を使う）:
1. VLMプロンプトの探索対象文字列
2. CLIPゼロショットの文言（プロンプトセット）
3. MOG2の動体サイズ範囲（min_area/max_area）

## ディレクトリ構成

```
configs/            # pipeline.yaml（共通既定値）+ targets/*.yaml（対象別設定）
data/labels/<target>/bin_labels.csv   # 人間確認済みビンラベル（ユーザーが配置。スキーマのみ用意）
src/                 # ステージ別スクリプト（src/直下にフラットに配置）
eval/                # 検証用ハーネス（現時点ではプレースホルダー）
scripts/             # end-to-endオーケストレーション
tests/
```

## 元になった資料

- `/Users/ktak0119/Documents/20260709_InsectVisitPipeline/PLAN.md` — 方針決定の経緯
- `/Users/ktak0119/Documents/2026624_ClaudeDetection/work/` — Stage1/3/5/6の元になった既存資産
- `/Users/ktak0119/Documents/20260331_bombusDetectTest/` — YOLOモデル・Stage5レビューツールの元
