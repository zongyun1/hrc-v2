"""Build a local navigation comparison report from completed native/Isaac runs."""
import argparse
import html
import json
import math
import os
from pathlib import Path


def summarize(result, native):
    if result.get("status") != "simulated" or result.get("task") != "NavigateKitchen":
        raise ValueError("Comparison requires a completed NavigateKitchen run from each backend")
    trace = result["trace"]
    states = trace if native else [t["robot"] for t in trace]
    times = [t["time_s"] for t in trace] if native else [(t["step"] + 1) * result.get("physics_dt", 1/120) for t in trace]
    successes = [t["source_success"] for t in trace] if native else [t["predicate"] for t in trace]
    return {"task_success": result["robot_task_success"], "final_distance_m": states[-1]["target_distance_m"],
            "final_orientation_cos": states[-1]["orientation_cos"],
            "first_success_s": next((t for t, success in zip(times, successes) if success), None),
            "last_sample_s": times[-1], "sample_count": len(trace),
            "base_path_length_m": sum(math.dist(a["base_pos"], b["base_pos"]) for a, b in zip(states, states[1:])),
            "initial_base_pos": states[0]["base_pos"], "final_base_pos": states[-1]["base_pos"],
            "physics_dt_s": result.get("physics_dt", None if native else 1/120),
            "control_hz": result.get("control_hz", None if native else 120),
            "camera_enabled": result.get("camera_enabled", False)}


def compare(native_path, isaac_path, output):
    native = json.loads(native_path.read_text())
    isaac = json.loads(isaac_path.read_text())
    if native.get("backend") != "original_robocasa_mujoco" or isaac.get("mode") != "original_pandaomron_navigation":
        raise ValueError("Expected original RoboCasa and migrated PandaOmron navigation results")
    params = native.get("navigation_parameters", {"speed_cap": .6, "waypoint_tolerance": .12})
    report = {"task": "NavigateKitchen", "native_result": str(native_path), "isaac_result": str(isaac_path),
              "instance": {key: native[key] for key in ("layout", "style", "seed", "target_fixture")},
              "native": summarize(native, True), "isaac": summarize(isaac, False),
              "native_predicate_agreement": native["predicate_agreement"],
              "native_reset_qpos_max_error": native["initial_qpos_max_error"],
              "native_controller_reset_checks": native.get("controller_reset_checks"),
              "native_navigation_parameters": params,
              "limitations": [
                  "Same exported task instance is used by the runners; the legacy Isaac result lacks a source XML hash, so this report cannot independently prove identical source snapshots.",
                  f"Original controllers, actuators and friction differ from the migrated controller. Native waypoint tolerance {params['waypoint_tolerance']} m and velocity cap {params['speed_cap']} m/s; Isaac .08 m and .3 m/s.",
                  "Timing/distance numbers describe these rollouts, not a simulator accuracy or speed ranking. Isaac physics timestep uses the recorded value when available; missing timestep/control rate use runner defaults.",
              ]}
    if not native.get("controller_reset_checks"):
        report["limitations"].append("Legacy native run: OSC reset frame was not checked; it may contain the known native-wrapper initialization bug. Do not treat its arm behavior as a clean original baseline.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    metrics = [("任务成功", "task_success"), ("最终距离 / m", "final_distance_m"),
               ("最终朝向误差余弦", "final_orientation_cos"), ("首次达标 / 仿真秒", "first_success_s"),
               ("底盘路径长度 / m", "base_path_length_m"), ("物理步长 / 秒", "physics_dt_s"),
               ("控制频率 / Hz", "control_hz")]
    def format_value(value):
        return f"{value:.6g}" if isinstance(value, float) else str(value)
    rows = "".join(f"<tr><th>{name}</th><td>{format_value(report['native'][key])}</td><td>{format_value(report['isaac'][key])}</td></tr>"
                   for name, key in metrics)
    videos = []
    for label, path, result in (("原版 RoboCasa / MuJoCo", native_path, native), ("迁移版 / Isaac Lab", isaac_path, isaac)):
        video = path.parent / "preview.mp4"
        src = html.escape(os.path.relpath(video.resolve(), output.resolve()), quote=True)
        media = f'<video controls preload="metadata" src="{src}"></video>' if result.get("camera_enabled") and video.exists() else '<p>本轮未保存录像</p>'
        videos.append(f"<section><h2>{label}</h2>{media}</section>")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoboCasa · Isaac Lab 迁移对照</title><style>
body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;padding:0 24px;max-width:1300px;background:#f6f7f9;color:#182333}
h1{font-size:28px}h2{font-size:18px}.videos{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;background:#111;border-radius:8px}
table{border-collapse:collapse;width:100%;background:white;margin:24px 0}td,th{text-align:left;padding:10px 16px;border-bottom:1px solid #ddd}
button{padding:8px 16px;margin-right:8px;cursor:pointer}input{width:300px;max-width:65vw}small{color:#536173}@media(max-width:800px){.videos{grid-template-columns:1fr}}
</style><h1>原版 RoboCasa 与 Isaac Lab 迁移对照</h1>
<p>NavigateKitchen · layout=LAYOUT · style=STYLE · seed=SEED</p>
<div class="videos">VIDEOS</div><p><button id="play">同时播放</button><button id="pause">暂停</button><button id="reset">回到起点</button>
<input id="seek" aria-label="共同时间轴" type="range" min="0" max="42" step="0.05" value="0"> <span id="time">0.00 s</span></p>
<small>按视频时间同步；两端使用不同控制器，动作进度不要求一致。</small>
<table><thead><tr><th>指标</th><th>原版 RoboCasa</th><th>Isaac Lab</th></tr></thead><tbody>ROWS</tbody></table>
<p>这是同一导出实例的任务级对照。原版使用 HYBRID_MOBILE_BASE / OSC，迁移版使用关节驱动；步长、驱动、摩擦近似及导航参数不同，因此不能直接将轨迹差异归为物理引擎误差，也不能把到达时间当作性能排名。</p>
<p>原版控制器初始化：RESET_STATUS。</p>
<p>原版逐步核对原判定与移植判定。旧版 Isaac 结果没有保存源 XML 哈希，当前报告无法独立验证两个历史文件的源快照完全相同。原版动作和完整 qpos/qvel 保存于对应目录的 rollout.npz。</p>
<p><a href="comparison.json">下载数值对照 JSON</a></p>
<script>
const vs=[...document.querySelectorAll('video')], seek=document.getElementById('seek'), time=document.getElementById('time');
function jump(t){vs.forEach(v=>{if(Number.isFinite(v.duration))v.currentTime=Math.min(t,v.duration)});seek.value=t;time.textContent=Number(t).toFixed(2)+' s'}
document.getElementById('play').onclick=()=>{vs.forEach(v=>v.play().catch(()=>{}))};
document.getElementById('pause').onclick=()=>vs.forEach(v=>v.pause());
document.getElementById('reset').onclick=()=>{vs.forEach(v=>v.pause());jump(0)};
seek.oninput=()=>jump(Number(seek.value));
vs.forEach(v=>v.addEventListener('loadedmetadata',()=>{const ds=vs.map(x=>x.duration).filter(Number.isFinite);if(ds.length)seek.max=Math.min(...ds)}));
if(vs.length)vs[0].addEventListener('timeupdate',()=>{seek.value=vs[0].currentTime;time.textContent=vs[0].currentTime.toFixed(2)+' s';
vs.slice(1).forEach(v=>{if(!vs[0].paused && !v.paused && Math.abs(v.currentTime-vs[0].currentTime)>.2)v.currentTime=Math.min(vs[0].currentTime,v.duration)})});
</script></html>'''
    for key in ("layout", "style", "seed"):
        page = page.replace(key.upper(), html.escape(str(native[key])))
    page = page.replace("VIDEOS", "".join(videos)).replace("ROWS", rows)
    page = page.replace("RESET_STATUS", "已验证目标坐标系一致" if native.get("controller_reset_checks") else "旧结果未验证，可能含已知 OSC 初始化错误")
    (output / "index.html").write_text(page)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--isaac", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    compare(args.native, args.isaac, args.output)
