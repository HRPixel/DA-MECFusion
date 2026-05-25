推荐使用方式
先为已有 processed 数据生成清单：

python tools/build_manifest.py --data_root data/processed/MSRS --splits train test
训练和测试命令可以保持原样，MSRSDataset 会自动读取 manifest：

python train.py --data_root data/processed/MSRS
python test.py --data_root data/processed/MSRS --split test --checkpoint experiments/damecfusion_v1_msrs/checkpoints/best.pth
评价阶段推荐新命令：

python eval.py --manifest data/processed/MSRS/splits/test.csv --fused_dir results/DA-MECFusion_V1/MSRS/fused --save_csv results/DA-MECFusion_V1/MSRS/metrics.csv --summary_txt results/DA-MECFusion_V1/MSRS/summary.txt
旧命令仍兼容：

python eval.py --ir_dir data/processed/MSRS/test/ir --vis_dir data/processed/MSRS/test/vis --fused_dir results/DA-MECFusion_V1/MSRS/fused