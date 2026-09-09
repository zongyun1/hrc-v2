"""Build a local, reproducible ServeTea failure comparison from recorded runs."""
import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, isaac, geometry = [json.loads(p.read_text()) for p in (args.native, args.isaac, args.geometry)]
    nr, ir = native["trace"], isaac["trace"]
    summary = {}
    replay_path = Path(native["replay"]["path"])
    if replay_path.exists():
        replay_rows = json.loads(replay_path.read_text())["trace"]
        count = min(len(replay_rows), len(ir))-1
        summary["replay_shared_prefix"] = {"samples": count, "identical": replay_rows[:count] == ir[:count]}
    for backend, rows in (("native", nr), ("isaac", ir)):
        positions = [(r.get("robot") or r)["object_pos"] for r in rows]
        summary[backend] = {"task_success": (native if backend == "native" else isaac)["robot_task_success"],
                            "max_cup_rise_mm": 1000*(max(p[2] for p in positions)-positions[0][2]),
                            "final_cup_displacement_mm": 1000*math.dist(positions[-1], positions[0])}
    summary["native"]["contact_geom_pairs"] = {" / ".join(k): v for k,v in Counter(
        tuple(c["geoms"]) for r in nr for c in r["gripper_cup_contacts"]).items()}
    summary["isaac"]["contact_sample_counts"] = dict(Counter(
        key for r in ir for key, value in r["serve_tea"]["contact_counts"].items() if value > 0))
    summary["isaac"]["failure"] = ir[-1]["robot"]["failure"]
    summary["geometry"] = {k:v for k,v in geometry.items() if k != "trace"}
    summary["native_tracking_position_error_m"] = {}
    for phase in ("pregrasp", "reach", "close", "lift"):
        errors = []
        for row in nr:
            if row["phase"] != phase:
                continue
            reference = min(ir, key=lambda r: abs(r["step"]*isaac["physics_dt"]-row["step"]/native["control_hz"]))
            errors.append(math.dist(row["tcp"], reference["robot"]["tcp"]))
        summary["native_tracking_position_error_m"][phase] = {"mean": sum(errors)/len(errors), "max": max(errors)}
    summary["limitations"] = "Native tracks Isaac achieved poses using original OSC, with different gripper/control dynamics. Tracking errors compare post-step native state to nearest input-time reference. Geometry check is static FK, not dynamics equivalence. Neither run is a native expert demonstration."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2))
    videos = [os.path.relpath(p.parent / "preview.mp4", args.output) for p in (args.native, args.isaac)]
    document = '''<!doctype html><meta charset="utf-8"><title>ServeTea 两端失败对照</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;background:#f7f8fa;color:#172033}.videos{display:flex;gap:20px}.videos>div{width:50%}video{width:100%}button,input{margin:12px}pre{white-space:pre-wrap;background:white;padding:20px}</style>
<h1>ServeTea：抓取未形成双指夹持</h1>
<p>相同 layout=2 / style=1 / seed=0。原版 OSC 跟踪迁移版实际末端轨迹；两边均未抓起杯子。此实验不是原版专家演示，也不是相同控制器下的物理引擎等价测试。</p>
<div class="videos"><div><h2>原版 RoboCasa / MuJoCo</h2><video id="native" controls src="NATIVE"></video></div><div><h2>Isaac Lab 复现</h2><video id="isaac" controls src="ISAAC"></video></div></div>
<button id="play">同时播放</button><button id="pause">暂停</button><label>共同时间 <input id="seek" type="range" min="0" max="17.4" step="0.05" value="0"><span id="time">0</span> 秒</label>
<p>重点查看 9–12.5 秒的合爪与抬升。原版只记录到左手指侧面接触杯子；Isaac 原有指垫接触计数为零。画面视角不同，请结合轨迹与接触数据判断。</p>
<p>原版重放输入与新复现的共同采样前缀检查见 replay_shared_prefix（排除末帧）；其余数值为原版运行及新 Isaac 复现。</p><pre>SUMMARY</pre>
<script>const vs=[document.querySelector('#native'),document.querySelector('#isaac')];document.querySelector('#play').onclick=()=>vs.forEach(v=>v.play());document.querySelector('#pause').onclick=()=>vs.forEach(v=>v.pause());document.querySelector('#seek').oninput=e=>{vs.forEach(v=>v.currentTime=+e.target.value);document.querySelector('#time').textContent=e.target.value};</script>'''
    document = document.replace("NATIVE", html.escape(videos[0], quote=True)).replace("ISAAC", html.escape(videos[1], quote=True)).replace("SUMMARY", html.escape(json.dumps(summary, indent=2, ensure_ascii=False)))
    (args.output / "index.html").write_text(document)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
