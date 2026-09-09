"""Compare a verified native action demonstration with its Isaac contact rollout."""
import argparse
import html
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, isaac = [json.loads(p.read_text()) for p in (args.native, args.isaac)]
    if native["mode"] != "official_demonstration_action_replay":
        raise ValueError("Expected original demonstration action execution")
    results = {}
    for key, data in (("native", native), ("isaac", isaac)):
        rows = data["trace"]
        cup = [(r.get("robot") or r)["object_pos"] for r in rows]
        success = [r["source_success"] if key == "native" else r["predicate"] for r in rows]
        times = [r["time_s"] if key == "native" else (r["step"]+1)*data["physics_dt"] for r in rows]
        results[key] = {"robot_task_success": data["robot_task_success"],
                        "source_predicate_at_end": data["source_predicate_at_end"],
                        "first_success_s": next((t for t,s in zip(times, success) if s), None),
                        "duration_s": times[-1], "cup_initial": cup[0], "cup_final": cup[-1],
                        "max_cup_rise_m": max(p[2] for p in cup)-cup[0][2],
                        "final_contacts": rows[-1]["contacts"] if key == "native" else rows[-1]["serve_tea"]["contacts"]}
        tilt = rows[-1].get('cup_tilt_degrees') if key == 'native' else rows[-1]['serve_tea'].get('cup_tilt_degrees')
        results[key]['final_cup_tilt_degrees'] = tilt
        results[key]['upright_task_success'] = (bool(data['robot_task_success'] and tilt <= 10.) if key == 'native' and tilt is not None
                                                else data.get('upright_task_success'))
    results["final_cup_backend_distance_m"] = math.dist(results["native"]["cup_final"], results["isaac"]["cup_final"])
    results["episode"] = native["episode"]
    results["scene"] = {k: native["episode_metadata"][k] for k in ("layout_id", "style_id")}
    results["placement_feedback"] = isaac.get("demonstration_feedback", False)
    results["scope"] = "Native executes official recorded actions. Isaac follows verified native joint motion and gripper targets; when placement_feedback is true, live cup/saucer positions retarget the final EEF trajectory. Both use real contacts; controllers and timesteps differ."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2))
    cards = []
    for key, label, path in (("native", "原版 RoboCasa：官方动作执行", args.native), ("isaac", "Isaac Lab：成功轨迹驱动", args.isaac)):
        video = html.escape(os.path.relpath(path.parent / "preview.mp4", args.output), quote=True)
        status = ("直立放置通过" if results[key]['upright_task_success'] else
                  "直立放置未通过" if results[key]['upright_task_success'] is False else "直立姿态未验证")
        cards.append(f'<section><h2>{label} · {status}</h2><video controls src="{video}"></video></section>')
    page = '''<!doctype html><meta charset="utf-8"><title>ServeTea 成功示范迁移</title>
<style>body{font:16px system-ui;margin:32px auto;padding:0 24px;max-width:1400px;background:#f5f7fa;color:#192539}.videos{display:flex;gap:20px}section{width:50%}video{width:100%}h2{font-size:19px}button,input{margin:12px}pre{white-space:pre-wrap;background:white;padding:20px}</style>
<h1>ServeTea 成功示范迁移</h1><p>相同示范场景与初态；原版执行官方动作，Isaac 通过关节驱动跟随成功轨迹。若下方 placement_feedback 为 true，放杯阶段另根据实时杯碟位置修正末端目标。两端均依赖真实接触，不在执行期间设置杯子位姿。控制器和物理步长不同。</p>
<div class="videos">CARDS</div><button id="play">同时播放</button><button id="pause">暂停</button><label>时间 <input id="seek" type="range" min="0" max="DURATION" step="0.05" value="0"><span id="time">0</span> 秒</label>
<pre>RESULTS</pre><script>const vs=[...document.querySelectorAll('video')];document.querySelector('#play').onclick=()=>vs.forEach(v=>v.play());document.querySelector('#pause').onclick=()=>vs.forEach(v=>v.pause());document.querySelector('#seek').oninput=e=>{vs.forEach(v=>v.currentTime=Math.min(+e.target.value,v.duration||+e.target.value));document.querySelector('#time').textContent=e.target.value};</script>'''
    page = page.replace("CARDS", "".join(cards)).replace("DURATION", str(max(results[k]["duration_s"] for k in ("native", "isaac")))).replace("RESULTS", html.escape(json.dumps(results, ensure_ascii=False, indent=2)))
    (args.output / "index.html").write_text(page)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
