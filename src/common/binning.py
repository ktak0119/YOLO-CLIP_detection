"""固定長ビンの割当・境界計算。5秒固定ビン（仕様書§1）を全ステージで共通に扱うための唯一の実装。"""


def bin_id_for(t_sec: float, bin_sec: float) -> int:
    return int(t_sec // bin_sec)


def bin_bounds(bin_id: int, bin_sec: float, video_duration_sec: float = None):
    start = round(bin_id * bin_sec, 2)
    end = round(start + bin_sec, 2)
    if video_duration_sec is not None:
        end = min(end, round(video_duration_sec, 2))
    return start, end
