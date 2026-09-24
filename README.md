# DA-MECFusion V2

本目录是 DA-MECFusion 当前代码仓库，实现复杂退化感知的红外—可见光图像融合。当前主版本为 V2，以 FLAME3 为主数据集，包含三类显式 prior、MEC 模态权重、训练、推理、评价与消融诊断工具；保留 V1/MSRS 接口兼容性。

2026-09-23代码快照作为`main`主版本，使用`v2-main-20260923`标签固定。仓库只发布代码和使用说明，不包含数据集、checkpoint、实验输出、人工标注、冻结实验清单或论文内部证据。下文V1命令和V2-Base命令用于接口说明，不等同于正式A0实验的完整复现包；运行时需自行准备配对数据、序列清单与模型权重。

## 当前方法流程

```text
红外/可见光输入
→ 规则型 prior
→ CNN 编码器
→ MEC 空间动态权重
→ 融合解码器
→ 融合图像
```

张量布局为 `[B, C, H, W]`，图像张量取值范围为 `[0, 1]`。

## 代码结构

| 路径 | 用途 |
|---|---|
| `train.py` | MSRS训练入口，保存checkpoint、日志和调试图 |
| `infer.py` | 单对红外/可见光图像推理 |
| `test.py` | MSRS批量测试并保存融合图、prior和权重 |
| `eval.py` | 对已有融合结果计算指标并生成CSV和摘要 |
| `models/` | prior、编码器、MEC、解码器、完整模型和loss |
| `datasets/` | MSRS读取、预处理和CSV索引工具 |
| `metrics/` | 融合指标实现 |
| `utils.py` | 运行目录、图像、可视化和日志共用工具 |
| `data/` | 代码保留的数据布局示例，当前不作为原始数据主存储位置 |
| `runs/` | 代码运行时生成的本地输出，不纳入Git仓库 |

## 环境

```powershell
python -m pip install -r requirements.txt
```

主要依赖包括 PyTorch、NumPy、Pillow、Matplotlib、OpenCV、SciPy和Pandas。实际使用CUDA前还应确认PyTorch版本与本机CUDA环境匹配。

## 数据位置

原始数据集统一保存在项目外部的[`../../datasets/`](../../datasets/)目录。当前代码参数默认值仍为`data/MSRS`，从本目录运行时应显式传入外部MSRS路径：

```text
../../datasets/MSRS/
├─ train/
│  ├─ ir/
│  ├─ vis/
│  └─ label/
├─ test/
│  ├─ ir/
│  ├─ vis/
│  └─ label/
└─ splits/
   ├─ train.csv
   └─ test.csv
```

CSV索引格式：

```csv
name,ir,vis,label
00001D,train/ir/00001D.png,train/vis/00001D.png,train/label/00001D.png
```

## 运行命令

以下命令均从本目录执行。

训练：

```powershell
python train.py --data_root ../../datasets/MSRS --run_root runs --run_name DA-MECFusion_v1_MSRS
```

单对推理：

```powershell
python infer.py --ir_path ../../datasets/MSRS/test/ir/00001.png --vis_path ../../datasets/MSRS/test/vis/00001.png --checkpoint runs/<训练目录>/checkpoints/best.pth
```

批量测试：

```powershell
python test.py --data_root ../../datasets/MSRS --checkpoint runs/<训练目录>/checkpoints/best.pth
```

指标评估：

```powershell
python eval.py --manifest ../../datasets/MSRS/splits/test.csv --data_root ../../datasets/MSRS --fused_dir runs/<测试目录>/outputs/fused
```

`test.py`负责生成图像；定量指标由`eval.py`生成。

`eval.py`默认使用`--metric_profile legacy`，保持历史CSV列`EN, SD, AG, MI, Qabf`不变；其中`Qabf`是V1 Sobel梯度相关近似量，不是标准\(Q^{AB/F}\)。论文复评使用：

```powershell
python eval.py --manifest <split.csv> --data_root <dataset_root> --fused_dir runs/<测试目录>/outputs/fused --run_root runs --run_name eval_standard --metric_profile standard
```

`standard`输出`EN, SD, AG, MI, SCD, Qabf_standard`。`MI`定义为`MI(F,IR)+MI(F,VIS)`；`SCD`采用Aslantas–Bendes差分相关和；`Qabf_standard`采用Xydeas–Petrovic边缘信息传递公式及常用参数`L=1, Tg=0.9994, kg=-15, Dg=0.5, Ta=0.9879, ka=-22, Da=0.8`。两种Qabf列名不得混用。

## V2-Base FLAME3 接口

V2-Base 复用现有模型主体和通用 manifest 读取，不复制一套 FLAME3 dataset 类。正式划分必须按完整序列指定，不能随机逐图切分。

生成序列级 manifest（序列编号必须在可视审计后替换并冻结）：

```powershell
python tools/flame3_make_splits.py --inventory runs/evidence_v2_flame3_20260705/data_audit/inventory.csv --output_dir runs/evidence_v2_flame3_20260705/splits_candidate --val_sequences SEQ_XX --test_sequences SEQ_YY --calibration_sequences SEQ_ZZ
```

脚本在未显式给出 `--train_sequences` 时，将未被 validation/test/calibration 占用的完整序列分配到 train，并拒绝序列重叠、未知序列和空的 train/validation/test。

V2-Base 训练：

```powershell
python train.py --data_root "../../datasets/FLAME 3  CV Dataset (Sycan Marsh)" --train_manifest runs/evidence_v2_flame3_20260705/splits_candidate/train.csv --val_manifest runs/evidence_v2_flame3_20260705/splits_candidate/validation.csv --run_root runs --run_name DA-MECFusion_v2_base_FLAME3 --epochs 50 --batch_size 4 --lr 0.0001 --height 512 --width 640 --num_workers 4 --device cuda --feature_channels 64 --lambda_int 1.0 --lambda_grad 2.0 --lambda_thermal 1.0 --lambda_sat 0.5 --gradient_mode directional --smoke_prior_mode base --prior_residual_scale 0 --seed 42
```

`best.pth` 在提供 `--val_manifest` 时按最低 validation total loss 保存；`run_config.json`、`train_log.csv` 和 `val_log.csv` 同时记录运行配置与损失。

V2 批量测试：

```powershell
python test.py --data_root "../../datasets/FLAME 3  CV Dataset (Sycan Marsh)" --manifest runs/evidence_v2_flame3_20260705/splits_candidate/test.csv --checkpoint runs/<V2训练目录>/checkpoints/best.pth --run_root runs --run_name test_FLAME3 --height 512 --width 640 --device cuda
```

V1 命令保持兼容：未传 `--gradient_mode` 时仍使用原幅值梯度，未传 `--intensity_target` 时仍使用逐像素 `max(IR,VIS-Y)` 强度目标，未传 `--val_manifest` 时仍按训练损失选 checkpoint，未传 `--smoke_prior_mode` 时仍使用 V1 的 `base` prior，未传 `--prior_residual_scale` 时仍为 `0`，未传 `--bounded_learned_gap` 时仍使用原始 learned logits，未传 `--equalize_feature_magnitude` 时仍直接融合原 encoder 特征。V2-MEC-1 通过非零 `prior_residual_scale` 把三类中心化 prior 的平均值以正负相反方向加入 IR/VIS logits。V2-MEC-2 额外使用 `--bounded_learned_gap`，按 `d_b=2*tanh(d/2)` 将 learned IR−VIS logit gap 限制在 `(-2,2)` 后再叠加 residual。V2-MEC-3 候选使用 `--equalize_feature_magnitude`，只在最终加权融合前把两路特征调整到相同的逐像素通道平均绝对幅值，并以两路原幅值均值作为共同尺度；MEC 权重预测仍读取原 encoder 特征。V2-Loss-1 使用 `--intensity_target mec_weighted`，把强度目标改为停止梯度的 `w_ir*IR+w_vis*VIS-Y`，只改变训练监督，不改变 test/infer 前向结构。训练 checkpoint 会记录这些模式；`test.py`、`infer.py` 与机制诊断默认从 checkpoint 恢复模型模式，只有显式传参时才覆盖。

V2内部消融仅使用四个默认关闭的训练参数：`--disable_thermal_prior`、`--disable_sat_uncertainty`和`--disable_smoke_prior`只阻止相应prior进入MEC，原始prior图及thermal/saturation损失保持不变；`--fixed_equal_weights`在训练和推理中固定`w_ir=w_vis=0.5`。这些模式写入checkpoint，`test.py`、`infer.py`和机制诊断自动恢复，不提供测试阶段的消融模式覆盖参数。

工作站首次运行前执行最小自检：

```powershell
python tools/flame3_make_splits.py --self_check
python models/losses.py
python models/priors.py
python models/mec_module.py
python models/damecfusion.py
python tools/check_v2_ablation_modes.py
python metrics/fusion_metrics.py
python train.py --help
python test.py --help
python infer.py --help
python eval.py --help
```

## 运行输出

运行脚本在`runs/`下创建带时间戳的目录，主要包含：

```text
checkpoints/
outputs/
├─ fused/
├─ priors/
├─ weights/
└─ comparisons/
logs/
metrics/
```

`runs/`用于单次代码运行，不等同于研究项目根目录中的[`../实验/`](../实验/)和[`../结果/`](../结果/)。完成运行后再根据研究需要整理实验记录和保留结果。

## 当前边界

- 当前不连接TNO、M3FD、FMB或RoadScene；FLAME3 通过显式序列 manifest 接入 V2-Base。
- 当前已实现V2内部消融与区域诊断工具；正式研究结果单独归档，不随代码发布。外部baseline代码不包含在本仓库。
- 正式FLAME3序列划分已在研究资产中冻结；本仓库不分发该清单，不能直接把README中的占位序列用于正式训练。
