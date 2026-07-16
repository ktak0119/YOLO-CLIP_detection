#!/usr/bin/env python3
"""Stage5: 人間目視フィルタリング（ビン単位目視スクリーニングツール）。

20260331_bombusDetectTest/pipeline/screening.py を移植・改修したもの。
「代表画像を1件ずつ表示→Y/→でTP、N/←でFP、Zで1件戻る、Qで終了」というUIロジックは
無改修。入力をbouts.csv+ファイル名globマッチングから、ビン単位テーブル（Stage4出力等の
video, bin_id, start_sec, end_sec, crop_path を含むCSV）に差し替え、crop_path列を
直接使うようにした。

キー操作:
    Y / →  : TP（陽性・保持）
    N / ←  : FP（陰性・除外）
    Z      : 1つ前に戻る
    Q      : 中断（途中まで保存済み、再実行で続きから再開）
"""
import argparse
import csv
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

OUTPUT_FIELDS = ["video", "bin_id", "start_sec", "end_sec", "crop_path", "screening"]


def load_bins(bin_table_csv: Path):
    """スコア（combined_score優先、無ければclip_score）の降順に並べ替えて返す。
    Stage4を経由していない入力（stage4_build_bin_table.py出力をそのまま渡す等）でも、
    別途ソートする手間なくスコア上位から目視できるようにするため（仕様書§4）。"""
    with open(bin_table_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    score_col = "combined_score" if rows and "combined_score" in rows[0] else "clip_score"
    rows.sort(key=lambda r: -(float(r[score_col]) if r.get(score_col) not in (None, "") else -1))
    return rows


def save_results(output_path: Path, results):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)


class ScreeningApp:
    def __init__(self, root, bins, output_path):
        self.root = root
        self.bins = bins
        self.output_path = output_path
        self.results = []
        self.index = 0

        self.root.title("Bin Screening")
        self.root.configure(bg="#2b2b2b")
        self.root.bind("<Key>", self.on_key)

        self.image_label = tk.Label(root, bg="#2b2b2b")
        self.image_label.pack(pady=10)

        self.info_label = tk.Label(root, text="", font=("Helvetica", 12),
                                    fg="white", bg="#2b2b2b", justify="left")
        self.info_label.pack(pady=5)

        self.progress_label = tk.Label(root, text="", font=("Helvetica", 11),
                                        fg="#aaaaaa", bg="#2b2b2b")
        self.progress_label.pack(pady=2)

        def _btn(parent, text, command, bg, width=16):
            lbl = tk.Label(parent, text=text, bg=bg, fg="white",
                           font=("Helvetica", 12), width=width,
                           relief="raised", cursor="hand2", pady=6)
            lbl.bind("<Button-1>", lambda e: command())
            lbl.bind("<Enter>", lambda e: lbl.config(relief="sunken"))
            lbl.bind("<Leave>", lambda e: lbl.config(relief="raised"))
            lbl.bind("<ButtonRelease-1>", lambda e: lbl.config(relief="raised"))
            return lbl

        btn_frame = tk.Frame(root, bg="#2b2b2b")
        btn_frame.pack(pady=10)
        _btn(btn_frame, "✓  TP  (Y / →)", lambda: self.judge("TP"), "#4caf50").pack(side="left", padx=10)
        _btn(btn_frame, "✗  FP  (N / ←)", lambda: self.judge("FP"), "#f44336").pack(side="left", padx=10)
        _btn(btn_frame, "↩  戻る  (Z)", self.go_back, "#888888", width=14).pack(side="left", padx=10)

        self.show_current()

    def show_current(self):
        if self.index >= len(self.bins):
            self.finish()
            return

        b = self.bins[self.index]
        img_path = Path(b["crop_path"])

        if img_path.exists():
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.tk_img, text="")
        else:
            self.image_label.config(image="", text="画像なし", fg="gray")

        score = b.get("combined_score") or b.get("clip_score") or ""
        info = (f"Video : {b['video']}\n"
                f"Bin   : {b['bin_id']}  |  "
                f"Time  : {b['start_sec']} - {b['end_sec']} sec\n"
                f"Score : {score}")
        self.info_label.config(text=info)
        self.progress_label.config(
            text=f"{self.index + 1} / {len(self.bins)}  （判定済み: {len(self.results)}）"
        )

    def judge(self, result):
        b = self.bins[self.index]
        self.results.append({
            "video": b["video"],
            "bin_id": b["bin_id"],
            "start_sec": b["start_sec"],
            "end_sec": b["end_sec"],
            "crop_path": b["crop_path"],
            "screening": result,
        })
        save_results(self.output_path, self.results)
        self.index += 1
        self.show_current()

    def go_back(self):
        if self.index > 0 and self.results:
            self.results.pop()
            self.index -= 1
            save_results(self.output_path, self.results)
            self.show_current()

    def on_key(self, event):
        if event.keysym in ("y", "Y", "Right"):
            self.judge("TP")
        elif event.keysym in ("n", "N", "Left"):
            self.judge("FP")
        elif event.keysym in ("z", "Z"):
            self.go_back()
        elif event.keysym in ("q", "Q"):
            self.root.destroy()

    def finish(self):
        self.progress_label.config(text="全件完了！ウィンドウを閉じてください。", fg="#4caf50")
        self.image_label.config(image="", text="")
        self.info_label.config(text="")


def main():
    parser = argparse.ArgumentParser(description="Bin screening tool (Stage5)")
    parser.add_argument("--bin-table", required=True,
                         help="video,bin_id,start_sec,end_sec,crop_pathを含むCSV"
                              "（Stage4 apply_score_model.py出力等。事前にスコア順ソート推奨）")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bin_table_csv = Path(args.bin_table)
    output_path = Path(args.output)

    bins = load_bins(bin_table_csv)

    start_index = 0
    existing_results = []
    if output_path.exists():
        with open(output_path, encoding="utf-8-sig") as f:
            existing_results = list(csv.DictReader(f))
        start_index = len(existing_results)
        print(f"再開: {start_index} 件スキップ")

    root = tk.Tk()
    app = ScreeningApp(root, bins, output_path)
    app.index = start_index
    app.results = existing_results
    app.show_current()
    root.mainloop()


if __name__ == "__main__":
    main()
