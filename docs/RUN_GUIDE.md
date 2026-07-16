# パイプライン運用手順書

## 全体の考え方

`configs/targets/<name>.yaml` は「対象(species)ごとの設定」であると同時に、実質的には
**「解析ジョブ（1回の動画セットの解析）ごとの設定」**として使う。新しい動画セットを解析する
たびに、既存のyamlをコピーして`videos_dir`/`out_dir`を書き換えるのが基本の使い方。
昆虫種が同じなら、モデルやプロンプトはそのまま使い回してよい（花の種類・撮影地が変わっても
対象昆虫が同じならconfig新規作成は不要——実例: bombusモデルをヘラナレン動画にそのまま適用した）。

## configファイルの全体像

```
configs/
  pipeline.yaml        # 全ジョブ共通の既定値（通常は書き換えない）
  targets/
    bombus.yaml         # ジョブ設定その1（コピーして使う雛形も兼ねる）
    butterfly.yaml       # ジョブ設定その2
    <あなたの新しいジョブ名>.yaml  # 新規解析のたびに追加
```

`--target <name>` を指定すると、`pipeline.yaml`（共通）+ `targets/<name>.yaml`（ジョブ別）が
マージされて使われる。**2つのファイルで同じキーを重複させるとエラーになる**
（`targets/*.yaml`に書いてよいのは対象固有の値だけ、という制約を機械的に強制するため）。

### `configs/pipeline.yaml`（全ジョブ共通、通常は触らない）

| パラメータ | 意味 |
|---|---|
| `bin_sec` | 固定ビン長（秒）。既定5.0 |
| `mog2_enabled` | Stage1(MOG2)を使うか。falseならStage2が全フレームに直接YOLOをかける |
| `mog2_history` / `mog2_var_threshold` / `mog2_learning_rate` / `mog2_warmup_frames` / `mog2_mask_top_frac` | MOG2背景差分の内部パラメータ（通常は既定値のままでよい） |
| `yolo_conf_threshold` | YOLO検知の信頼度閾値（1フレーム最大1件のbest boxを採用する基準） |
| `top_k_per_bin` | 1ビンあたり何枚を代表クロップとして残すか（K） |
| `yolo_batch_size` | YOLOバッチ推論のバッチサイズ |
| `tight_crop_pad_frac` / `tight_crop_pad_min_px` | クロップ余白の設定 |
| `clip_model` / `clip_pretrained` | 使用するCLIPモデル |
| `clip_apply_cutoff` | Stage3で足切りするか（仕様上false固定） |
| `target_recall` | Stage4の閾値決定で狙うRecall |
| `score_fusion_methods` | Stage4で比較する4方式のリスト |
| `clip_pad_sec` | Stage6のクリップ前後パディング秒数 |
| `merge_gap_sec` | Stage6でビンを結合する間隔しきい値 |
| `resume` | 既存出力があればスキップするか |

### `configs/targets/<name>.yaml`（ジョブごとに用意）

| パラメータ | 意味 |
|---|---|
| `target` | このジョブの名前（`--target`に渡す文字列と一致させる） |
| `videos_dir` | **解析対象のmp4が入っているディレクトリ**（新規ジョブごとに書き換える。必須） |
| `out_dir` | **Stage1-3の出力先ベースディレクトリ**（動画ごとに`out_dir/<動画名>/`が作られる）。`null`のままなら自動で`output/<target>/`（リポジトリ内、`.gitignore`対象）になる |
| `yolo_model_path` | 使用するYOLOモデルの`.pt`ファイルパス |
| `clip_prompt_key` | Stage3のCLIPプロンプトセットのキー（`stage3_clip_score.py`のPROMPT_SETS参照） |
| `vlm_search_target` | Stage3(任意VLM)/Stage5表示用の「探索対象」文字列 |
| `motion_min_area` / `motion_max_area` | MOG2の動体サイズ範囲（ピクセル²、640px幅リサイズ後基準）。チューニング手順は[motion_size_tuning.md](motion_size_tuning.md) |

`videos_dir`/`out_dir`はコマンドラインの`--videos-dir`/`--out-dir`でも上書きできる
（優先順位: CLI > configの値 > `out_dir`のみ`output/<target>/`への自動フォールバックあり）。
`videos_dir`は自動フォールバックが無いため、CLIかconfigのどちらかで必ず指定する必要がある。

## 新しい動画セットを解析する手順

### 1. ジョブ用configを用意する

```bash
cd /Users/ktak0119/Documents/YOLO-CLIP_detection_test   # または開発用クローン
cp configs/targets/bombus.yaml configs/targets/<新しいジョブ名>.yaml
```

`<新しいジョブ名>.yaml`を開いて`videos_dir`を書き換える（必須）。`out_dir`は`null`のままでよければ
`output/<新しいジョブ名>/`に自動出力される（以下の手順では`<out_dir>`と書いている箇所は
明示的に指定した場合はその値、指定していなければ`output/<新しいジョブ名>/`と読み替える）。
対象昆虫がbombusと違うなら`yolo_model_path`/`clip_prompt_key`/`vlm_search_target`も変える。
動体サイズを調整したいなら`motion_min_area`/`motion_max_area`も（詳細は上の表を参照）。

### 2. Stage1〜3を一括実行（バッチ）

```bash
caffeinate -i .venv/bin/python3 scripts/run_stage1_3_batch.py --target <新しいジョブ名>
```

`videos_dir`内の全mp4に対しStage1(MOG2フラグ)→Stage2(YOLO+ビン代表抽出)→Stage3(CLIPスコア)を
順に実行し、`out_dir/<動画名>/`に`stage1.json`・`stage2_bins.csv`・`crops/`・`stage3_scored.csv`
を書き出す。既に出力があるものは自動でスキップされる（resume）ので、中断しても再実行でよい。
長時間かかる場合は`caffeinate -i`をつけてスリープを防ぐ。

### 3. 全動画のStage3出力を1つのビンテーブルにまとめる

```bash
.venv/bin/python3 src/stage4_build_bin_table.py \
  --stage3-glob "<out_dir>/*/stage3_scored.csv" \
  --target <新しいジョブ名> \
  --out-csv <out_dir>/combined_bin_table.csv
```

### 4a. （精度を追い込みたい場合）Stage4でスコア統合モデルを決める

十分な数の目視ラベル（`data/labels/<target>/bin_labels.csv`、実際に目視して作る）があるなら:

```bash
.venv/bin/python3 src/stage4_fit_score_model.py \
  --bin-table <ラベル付きbin-table> --target <ジョブ名> --out-dir <out_dir>/stage4_fit
.venv/bin/python3 src/stage4_apply_score_model.py \
  --bin-table <out_dir>/combined_bin_table.csv \
  --models-json <out_dir>/stage4_fit/models.json \
  --method logistic_regression \
  --target <ジョブ名> --out-csv <out_dir>/stage4_applied.csv
```

ラベルがまだ無ければ、Stage4を飛ばして`combined_bin_table.csv`をCLIPスコア順に並べ替えて
そのままStage5に進んでよい（後述）。

### 5. Stage5: 人間目視

```bash
# CLIPスコア降順に並べ替えてから見る（任意、順位帯ごとに確認する仕様§4に沿う）
.venv/bin/python3 src/stage5_screening.py \
  --bin-table <combined_bin_table.csv または stage4_applied.csv> \
  --output <out_dir>/screening_result.csv
```

`Y`/`→`=TP、`N`/`←`=FP、`Z`=1つ戻る、`Q`=中断（続きから再開可能）。

### 6. Stage6: ビン結合＋クリップ出力

```bash
.venv/bin/python3 src/stage6_merge_bins.py \
  --screening-csv <out_dir>/screening_result.csv \
  --target <ジョブ名> \
  --out-json <out_dir>/merged.json
# --videos-dir を省略するとconfigのvideos_dirが使われる

.venv/bin/python3 src/stage6_make_clips.py \
  --manifest <out_dir>/merged.json \
  --target <ジョブ名> \
  --out-dir <out_dir>/clips
```

## 出力ディレクトリ構成（1ジョブ分）

```
<out_dir>/
  <動画名1>/
    stage1.json          # MOG2フラグ結果
    stage2_bins.csv       # YOLO検知+ビン代表クロップの一覧
    crops/                 # ビン代表クロップ画像
    stage3_scored.csv       # +CLIPスコア
    run_manifest.json        # 実行時のconfig+git sha記録
  <動画名2>/
    ...
  combined_bin_table.csv   # 全動画分をまとめたビンテーブル（Stage4-a後）
  stage4_applied.csv        # スコア統合適用後（Stage4使用時）
  screening_result.csv       # 人間目視結果（Stage5後）
  merged.json                 # マージ済みクリップマニフェスト（Stage6-a後）
  clips/                       # 最終クリップ動画（Stage6-b後）
```
