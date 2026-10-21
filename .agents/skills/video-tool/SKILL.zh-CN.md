# 视频处理工具 Skill（中文对照）

本文件是同目录 `SKILL.md` 的中文译本，供人阅读；Codex 只加载 `SKILL.md`。两者内容一致，如有出入以英文版为准。角色和每一轮的流程见 `AGENTS.md`；`RUN_DECISIONS.md` 优先于这里的默认规则。

## 组成

| 部分 | 位置 | 作用 |
|---|---|---|
| `clean-clear` | 每台服务器的工具目录 | 擦除字幕和台标；`--probe` 检查片源；`--check` 对比成片与片源 |
| worker | 每台服务器，supervisor 服务 `video-worker` | 从 `/workspace/queue/` 取任务，下载、预检、处理、检查、上传到暂存区、汇报 |
| `tools/ledger.py` | 项目目录（Sol 所在机器） | 台账、派片、复核命令、批次交付 |

需要 0.3.6 或更新的工具（worker 和 `--check`）。0.3.5 之前的版本对视频比音频晚开始的片源会重复首帧、丢掉最后一帧，非方形像素片源写成方形像素，视频起点归零；0.3.5 起已修正，所以绝不要再对它们的成片额外做起点重新封装。

## 位置

Google Drive，一个 rclone 远端 `gdrive:`，根目录是项目的 Drive 文件夹（rclone 配置里的 `root_folder_id`）：

```
input/                                   片源，陆续增加（文件名任意）
staging/<batch_id>/videos/<asset_id>.attempt<N>.mp4    等待复核的结果（及 subtitles/）
output/<batch_id>/                       交付：videos/、subtitles/、records_<batch_id>.csv、README.md
log/ledger.csv                           台账副本
log/servers/<主机>/heartbeat.json         每 5 分钟一次；每个任务一个 jobs/<asset_id>.json
log/qc/<asset_id>/attempt_<N>/           qc.json、sheet_full.jpg、sheet_detail.jpg、预检结果和日志
```

项目目录（Codex 运行的地方）：`AGENTS.md`、`RUN_DECISIONS.md`、`ops.json`（设置）、`assets.csv`（片源文件名或 Drive 文件 ID → asset_id、batch_id、work_order、options；按工作单填写）、`server-ssh-config`（每台服务器一个 `Host`，主机名即服务器编号）、`tools/ledger.py`、`state/`（台账、批次文件）、`review/`（下载下来的检查材料）、工具压缩包 `video_tool_<version>_with_propainter.tar.gz`。

服务器：工具在 `/workspace/video_tool_<version>/`，另有 `/workspace/worker.json`、`queue/ jobs/ done/ failed/`、`status.txt`、`heartbeat.json`、`health.json`、`worker.out.log`、`setup.log`。

## Luna：一台服务器

### 部署新服务器

`server-ssh-config` 里有、但没有心跳的主机就是新服务器。
1. `ssh -F server-ssh-config 主机 'nvidia-smi; df -h /workspace'`。要求：NVIDIA 显卡 16 GB 以上，RTX 50 系列的驱动支持 CUDA 12.8 或更新，可用空间 200 GB 以上。不满足就汇报并停止。
2. `python3 tools/ledger.py deploy 主机 video_tool_<version>_with_propainter.tar.gz`：复制 rclone 配置和压缩包，启动 `scripts/server_setup.sh`（系统软件、含 PyTorch 的安装、worker 服务）。需要 10–30 分钟。
3. 每隔几分钟看一次 `ssh ... tail -n 20 /workspace/setup.log`，直到出现 `== 主机 ready`。日志里必须有 `cuda available: True` 和 `CUDAExecutionProvider`；否则是驱动问题，汇报。
4. `ledger.py servers` 显示该主机有心跳即完成。worker 处理第一部影片时会先做 3 分钟的 GPU 健康检查。

### 检查服务器

读 `ssh ... cat /workspace/status.txt`（有异常时再看 `tail -n 30 /workspace/worker.out.log`）。一切正常时汇报一行：健康状态、当前影片和阶段、队列长度、剩余磁盘。

| 现象 | 含义 | 处理 |
|---|---|---|
| health 为 `unhealthy`，或 `degraded: True` | GPU 满载时达不到应有频率（主机限制） | 汇报"建议换机器"；其他什么都不做 |
| 心跳或 status 超过 15 分钟没更新，`supervisorctl status video-worker` 不是 RUNNING | worker 停了 | `supervisorctl start video-worker`；汇报 `worker.out.log` 最后几行 |
| 剩余磁盘低于 30 GB | 残留文件 | 只删除 `done/` 或 `failed/` 里任务对应的 `/workspace/jobs/*/attempt_*/in` 和 `out` |
| `failed/` 里有任务 | 看 `jobs/<asset_id>/attempt_<N>/state.json`（`error`）和 `run.log` | 汇报错误；是否重试由 Sol 决定 |
| `clean-clear` 还活着，但它的 `clean-clear.log` 20 分钟没有新行 | 工具没识别出的卡死 | 汇报；除非 Sol 要求，不要结束进程 |

worker 运行时不要手动启动 `clean-clear`（MPS 下第二个进程会退回 CPU）。只按进程号结束进程，不用 `pkill -f`。手动结束一次运行后：`echo quit | nvidia-cuda-mps-control`。

## Sol：复核每一部暂存结果

`ledger.py fetch ID` 把 `qc.json` 和对比图下载到 `review/ID/attempt_N/`。

1. `qc.json` → `checks`：每一项都是 `ok`。frames（帧数与片源相同）、resolution、frame_rate、aspect、colour、codec（H.264、MP4）、bitrate（不低于片源）、duration、av_start（视频相对音频的起点与片源一致）、audio、size、subtitles（SRT 有效）。另外看任务状态（`log/servers/<主机>/jobs/ID.json`）里的 `probe_warnings`。
2. `sheet_detail.jpg`：字幕区和台标区，左边片源、右边成片，8 个均匀分布的时间点加最暗的场景。查看：字幕的任何痕迹（笔画、轮廓、阴影）、台标的任何痕迹、模糊或拖影的色块、色斑、人脸或身体变形、影片本身文字（招牌、片名、演职员表）被损坏。`sheet_full.jpg`：整幅画面完好、没有错位。
3. 决定：
   - 都没问题：`approve ID --note ""`（备注会写进交付记录的 `qc_notes`；客户需要知道的事都写上，例如某一小段略软）。
   - 测量项不通过但有明确理由可以接受：只能加 `--force`，并在 `--note` 里写明理由。
   - 能修：`reject ID --note "..." --rerun "参数"`。字幕在上方：`--band 0,0.3`。台标没擦干净：`--watermark <掩码.png>`（已知掩码）。码率低于片源：`--crf 14`。字幕漏擦：`--min-score 0.5`。
   - 参数修不了，或第二次尝试仍失败：`reject ID --note "..."`（转为 needs_user），并告诉用户。
   - `no_output`（没找到字幕也没找到台标）：查看预检结果；影片确实没有需要擦的内容时，询问用户如何交付。

通过后，成片（和 SRT）会移入 `output/<batch_id>/` 并核对 MD5。

## Sol：批次

`ledger.py status` 显示每个批次的进度。一个批次的影片全部交付后：`ledger.py batch 批次号` 生成 `records_<batch_id>.csv`（附件二的 16 个字段）、交付字幕时的 `subtitles_<batch_id>.csv` 和 `README.md`，上传并核对。然后告诉用户该批次完成，附批次号、影片数和总时长。

## 手动使用工具（一次性任务）

`clean-clear 文件 -o 输出目录 [选项]`、`clean-clear --probe 文件 -o 目录`、`clean-clear --check 片源 成片 -o 目录`。选项：`--subtitles off`、`--band 0,0.3`、`--watermark auto|off|掩码.png`、`--crf 18`、`--jobs auto`。不要在 worker 正忙的服务器上手动运行。

## 禁止

- 修改、移动或覆盖原片和已交付成片；不看对比图就通过。
- 正式交付使用 `--cut`，除非工作单允许删减画面（合同要求与片源帧数相同）。
- 对有音频空洞的片源做音频变速；为了让结果通过而关闭检查。
- 把凭据写进日志、交付物或代码仓库。

## 已知限制（如实告诉用户）

- 只处理烧录的硬字幕和角落的静态台标。画面其他位置的说明文字、二维码、网址、滚动字幕不会擦除。
- 字幕行上的场景文字可能被一起擦掉；细节丰富背景上的填补可能变糊；快速运动穿过字幕区时可能轻微闪烁（ProPainter 的限制）。
- 不放大：低于 720p 的片源输出仍低于 720p。
- 正常的 RTX 5090 上：90 分钟 720p 影片带字幕约 1.3–2 小时，只擦台标约 40–55 分钟。
