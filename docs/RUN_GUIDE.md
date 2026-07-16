# パイプライン運用ガイド

## クイックスタート

```bash
# 1. ジョブ設定を作る（videos_dirだけ書き換えれば動く）
cp configs/job_template.yaml configs/targets/my_job.yaml
# my_job.yaml を開いて videos_dir を書き換える（他は既存ジョブの値を流用してもよい）

# 2. Stage1〜3を一括実行（長時間になりうるのでcaffeinateを推奨）
caffeinate -i .venv/bin/python3 scripts/run_stage1_3_batch.py --target my_job

# 3. 全動画の結果を1つの表にまとめる（スコア順ソート＋確認用画像フォルダを自動生成）
.venv/bin/python3 src/stage4_build_bin_table.py \
  --stage3-glob "output/my_job/*/stage3_scored.csv" \
  --target my_job \
  --out-csv output/my_job/combined_bin_table.csv

# 4. 人間目視（Stage4のスコア統合モデルが無くてもこのまま実行できる。スコア降順で表示される）
.venv/bin/python3 src/stage5_screening.py \
  --bin-table output/my_job/combined_bin_table.csv \
  --output output/my_job/screening_result.csv

# 5. 陽性ビンをクリップとして書き出す
.venv/bin/python3 src/stage6_merge_bins.py \
  --screening-csv output/my_job/screening_result.csv \
  --target my_job \
  --out-json output/my_job/merged.json
.venv/bin/python3 src/stage6_make_clips.py \
  --manifest output/my_job/merged.json \
  --target my_job \
  --out-dir output/my_job/clips
```

これで一通りのクリップが出力される。以下は各ステップの詳細と、設定ファイルの説明。

---

## 設定ファイルの考え方

1つのジョブ（1つの動画セットの解析）につき、`configs/targets/` に1つのyamlファイルを作る。
新規ジョブは必ず `configs/job_template.yaml` をコピーして作る（既存ジョブのyamlをコピーすると
「通常は触らない」パラメータまで一緒にコピーされてしまい、後で共通パラメータを見直すときに
どのジョブがどの値かわかりにくくなるため、テンプレートを起点にする）。

```
configs/
  job_template.yaml       # 新規ジョブ作成の起点（このファイル自体は直接使わない）
  targets/
    bombus.yaml            # 実際に使うジョブ設定（--target bombus で読み込まれる）
    butterfly.yaml
    my_job.yaml              # 新規ジョブを作るたびにここに追加
```

1ファイルの中は3つのセクションに分かれている:

| セクション | いつ書き換えるか |
|---|---|
| ① このジョブ固有の情報 | **毎回必ず**（`target`名・`videos_dir`・`out_dir`） |
| ② 対象昆虫ごとの設定 | 対象の昆虫種が変わるときだけ（モデルパス・CLIP文言・動体サイズ範囲） |
| ③ パイプライン共通パラメータ | 通常は触らない（挙動を細かく調整したいとき以外） |

同じ昆虫を対象にした別の動画セットを解析する場合、既存ジョブのyamlをコピーして
①だけ書き換えれば流用できる（実例: bombus用の設定を、別の花・別撮影地の動画にもそのまま適用した）。

---

## Step 1: ジョブ設定を作る

```bash
cp configs/job_template.yaml configs/targets/<ジョブ名>.yaml
```

`<ジョブ名>.yaml` を開いて書き換える:

- **`target`**: ジョブ名（ファイル名と合わせる）
- **`videos_dir`**: 解析対象のmp4が入っているディレクトリ（必須）
- **`out_dir`**: 出力先。`null`のままでよければ自動で`output/<ジョブ名>/`になる（推奨、通常はこれでよい）
- 対象昆虫が既存ジョブと違う場合のみ、セクション②（`yolo_model_path`・`clip_pos_prompts`・
  `clip_neg_prompts`・`vlm_search_target`・`motion_min_area`/`motion_max_area`）も書き換える

### CLIPプロンプトの書き方（`clip_pos_prompts`/`clip_neg_prompts`）

Stage3のCLIPは「pos文とneg文、どちらに画像が近いか」でスコアを付ける。**このスコアの質は
文言の質にかなり左右される**ので、以下を意識して書くと良い:

- 英語の単純な説明文にする（`"a photo of ..."` 形式。CLIPは英語自然文で学習されている）
- **posは複数、言い回し・状況を変えて用意する**（例: 止まっている／給餌している／羽を広げている／
  近接した状態、など。1文だけだと対象の見た目のバリエーションを拾い切れない）
- **negは実際の映像で誤検知しやすいものを具体的に**列挙する。定番は
  「風で揺れる葉」「ボケた花だけの写真」「光の反射（グレア）」「別の小さい虫やクモ」
  「動物がいない防犯カメラ映像」など（`job_template.yaml`に既定セットあり、そのまま流用可）
- **一度Stage3〜5を回してみて、スコアが高いのに間違っている画像（偽陽性）が見つかったら、
  その画像の見た目をそのまま言語化した文をnegに追加する**。これが一番効く。
  実例（butterfly.yamlに実際に入っている）: ある花の巻いた花弁と斑点模様が蝶の羽に見えて
  誤検知が多発していたため、`"a curled-back lily flower petal that looks like a wing but is
  just a petal"` のように**誤認の原因そのものを言葉にした文**を追加したところ、大きく改善した。
  「一般的なnegを増やす」より「今起きている誤検知の見た目をピンポイントで言語化する」方が効く。

### 動体サイズ範囲（`motion_min_area`/`motion_max_area`）

MOG2が拾う動体の面積範囲（ピクセル²）。対象が小さいと既定値で拾えないことがある。
詳しいキャリブレーション手順は [motion_size_tuning.md](motion_size_tuning.md) を参照。

---

## Step 2: Stage1〜3を一括実行

```bash
caffeinate -i .venv/bin/python3 scripts/run_stage1_3_batch.py --target <ジョブ名>
```

`videos_dir`内の全mp4に対し、動画ごとに以下を順に実行する:

- **Stage1**: MOG2で動体のあったフレームをフラグ（クリップは作らない）
- **Stage2**: フラグ済みフレームにYOLOをかけ、5秒ビンごとの代表クロップを抽出
- **Stage3**: 代表クロップにCLIPスコアを付与

出力は`out_dir/<動画名>/`に書かれる（既に出力があれば自動スキップ＝再開可能なので、
中断しても`caffeinate`ごと再実行してよい）。長時間かかる場合は`caffeinate -i`でスリープを防ぐ。

---

## Step 3: 全動画をまとめる

```bash
.venv/bin/python3 src/stage4_build_bin_table.py \
  --stage3-glob "<out_dir>/*/stage3_scored.csv" \
  --target <ジョブ名> \
  --out-csv <out_dir>/combined_bin_table.csv
```

全動画のStage3出力を1つの表に結合し、**CLIPスコアの高い順に並べ替える**。この並べ替えは
Step 4（Stage4を使うかどうか）の判断より前に、常に行われる——Stage4を使っても使わなくても、
Step 5の目視は必ずスコア順になる。**並べ替えておくと、目視の際に陽性がまとまって出てくるため
判定が捗る**（連続してTPが出る区間・連続してFPが出る区間ができ、パターンとして把握しやすい）。

同時に`<out_dir>/review_crops/` フォルダに、代表クロップ画像を`0001_score0.xxx_...jpg`のような
順位・スコア付きファイル名でコピーする。**Finderやプレビューでこのフォルダを名前順に
開けば、スコア上位から画像を一覧できる**（Stage5のGUIツールを使わず一覧性重視で
ざっと見たいときに便利）。候補数が非常に多い場合は`--skip-review-copy`でコピーを省略できる。

---

## Step 4（任意）: スコア統合モデルを決める

**目視ラベルがまだ無ければこのステップは飛ばしてよい**（Step 3の結果をそのままStep 5に渡せる）。

精度を追い込みたい場合、`data/labels/<target>/bin_labels.csv`に目視ラベル（動画/撮影日単位で
分割したもの、詳しくは[data/labels/README.md](../data/labels/README.md)）を用意した上で:

```bash
.venv/bin/python3 src/stage4_fit_score_model.py \
  --bin-table <ラベル付きbin-table> --target <ジョブ名> --out-dir <out_dir>/stage4_fit
.venv/bin/python3 src/stage4_apply_score_model.py \
  --bin-table <out_dir>/combined_bin_table.csv \
  --models-json <out_dir>/stage4_fit/models.json \
  --method logistic_regression \
  --target <ジョブ名> --out-csv <out_dir>/stage4_applied.csv
```

`stage4_applied.csv`ができたら、Step 5では`combined_bin_table.csv`の代わりにこちらを使う。

---

## Step 5: 人間目視

```bash
.venv/bin/python3 src/stage5_screening.py \
  --bin-table <out_dir>/combined_bin_table.csv \
  --output <out_dir>/screening_result.csv
```

（Step 4を実行した場合は`--bin-table`に`stage4_applied.csv`を指定する）

自動的にスコアの高い順に表示される（並べ替えは不要）。操作:

| キー | 意味 |
|---|---|
| `Y` / `→` | TP（陽性・保持） |
| `N` / `←` | FP（陰性・除外） |
| `Z` | 1つ前に戻る |
| `Q` | 中断（保存済み、再実行で続きから再開） |

---

## Step 6: ビン結合・クリップ出力

```bash
.venv/bin/python3 src/stage6_merge_bins.py \
  --screening-csv <out_dir>/screening_result.csv \
  --target <ジョブ名> \
  --out-json <out_dir>/merged.json

.venv/bin/python3 src/stage6_make_clips.py \
  --manifest <out_dir>/merged.json \
  --target <ジョブ名> \
  --out-dir <out_dir>/clips
```

TP判定されたビンのうち、時間的に隣接するものは1つのクリップにまとめられる
（意味的な訪花イベント統合ではなく機械的な区間結合）。最終的なmp4は`<out_dir>/clips/`に出力される。

---

## 設定パラメータ一覧（`configs/job_template.yaml`）

### ① このジョブ固有の情報（毎回書き換える）

| パラメータ | 意味 |
|---|---|
| `target` | ジョブ名。`--target`に渡す文字列 |
| `videos_dir` | 解析対象のmp4が入っているディレクトリ。必須 |
| `out_dir` | 出力先。`null`なら自動で`output/<target>/` |

### ② 対象昆虫ごとの設定（種が変わるときだけ）

| パラメータ | 意味 |
|---|---|
| `yolo_model_path` | 対象昆虫を検出するYOLOモデルの`.pt`パス |
| `vlm_search_target` | Stage5等に表示する「探索対象」の表示名（動作には影響しない） |
| `clip_pos_prompts` / `clip_neg_prompts` | Stage3 CLIPのゼロショット文言。書き方は上記参照 |
| `motion_min_area` / `motion_max_area` | Stage1(MOG2)が拾う動体の面積範囲（px²、640px幅基準） |

### ③ パイプライン共通パラメータ（通常は触らない）

| パラメータ | 意味 |
|---|---|
| `bin_sec` | 固定ビン長（秒）。既定5.0 |
| `mog2_enabled` | falseにするとStage1を使わずStage2が全フレームに直接YOLOをかける |
| `mog2_history`/`mog2_var_threshold`/`mog2_learning_rate`/`mog2_warmup_frames`/`mog2_mask_top_frac` | MOG2の内部パラメータ |
| `yolo_conf_threshold` | YOLO検知の信頼度閾値 |
| `top_k_per_bin` | 1ビンあたりの代表クロップ枚数（K） |
| `yolo_batch_size` | YOLOバッチ推論のバッチサイズ |
| `tight_crop_pad_frac`/`tight_crop_pad_min_px` | クロップ余白 |
| `clip_model`/`clip_pretrained` | 使用するCLIPモデル |
| `clip_apply_cutoff` | Stage3で足切りするか（仕様上false固定） |
| `target_recall` | Stage4の閾値決定で狙うRecall |
| `score_fusion_methods` | Stage4で比較する4方式 |
| `clip_pad_sec` | Stage6のクリップ前後パディング秒数 |
| `merge_gap_sec` | Stage6でビンを結合する間隔しきい値 |
| `resume` | 既存出力があればスキップするか |

`videos_dir`/`out_dir`はコマンドラインの`--videos-dir`/`--out-dir`でも上書きできる
（優先順位: CLI > configの値 > `out_dir`のみ`output/<target>/`への自動フォールバックあり）。

---

## 出力ディレクトリ構成

```
output/<target>/                    # out_dir を null のままにした場合のデフォルト
  <動画名1>/
    stage1.json                      # MOG2フラグ結果
    stage2_bins.csv                   # YOLO検知+ビン代表クロップの一覧
    crops/                             # ビン代表クロップ画像
    stage3_scored.csv                   # +CLIPスコア
    run_manifest.json                    # 実行時のconfig+git sha記録
  <動画名2>/
    ...
  combined_bin_table.csv               # 全動画分をまとめたビンテーブル（Step 3後、スコア降順）
  review_crops/                         # スコア順にリネームコピーされた確認用画像（Step 3後）
  stage4_fit/                            # モデル比較レポート（Step 4使用時）
  stage4_applied.csv                      # スコア統合適用後（Step 4使用時）
  screening_result.csv                     # 人間目視結果（Step 5後）
  merged.json                               # マージ済みクリップマニフェスト（Step 6前半後）
  clips/                                     # 最終クリップ動画（Step 6後半後）
```
