import importlib


def _load_rlds():
    # 可选依赖：没有 rlds 也能运行。
    try:
        return importlib.import_module("rlds")
    except Exception:  # pragma: no cover - optional dependency
        return None


rlds = _load_rlds()


def rlds_keys():
    if rlds is None:
        return {
            "steps": "steps",
            "observation": "observation",
            "action": "action",
            "reward": "reward",
            "is_first": "is_first",
            "is_last": "is_last",
            "is_terminal": "is_terminal",
            "episode_metadata": "episode_metadata",
        }
    return {
        "steps": rlds.STEPS,
        "observation": rlds.OBSERVATION,
        "action": rlds.ACTION,
        "reward": rlds.REWARD,
        "is_first": rlds.IS_FIRST,
        "is_last": rlds.IS_LAST,
        "is_terminal": rlds.IS_TERMINAL,
        "episode_metadata": rlds.EPISODE_METADATA,
    }


def build_rlds_episode(steps, metadata):
    # 组装一个最小的 RLDS episode 字典。
    keys = rlds_keys()
    return {
        keys["steps"]: steps,
        keys["episode_metadata"]: metadata,
    }
