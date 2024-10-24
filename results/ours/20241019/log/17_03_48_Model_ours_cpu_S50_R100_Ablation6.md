```python
|2024-10-19 17:03:48| {
     'Ablation': 6, 'bs': 4, 'classification': False, 'dataset': cpu,
     'debug': True, 'decay': 0.001, 'density': 0.8, 'device': cpu,
     'device_name': core-i7-7820x, 'epochs': 1, 'eval_device': desktop-cpu-core-i9-13900k-fp32, 'experiment': 0,
     'graph_encoder': gat, 'heads': 8, 'llm': 1, 'log': <utils.logger.Logger object at 0x7fd988070310>,
     'logger': None, 'loss_func': L1Loss, 'lr': 0.001, 'model': ours,
     'op_encoder': one_hot, 'optim': AdamW, 'order': 4, 'path': ./datasets/,
     'patience': 100, 'program_test': False, 'rank': 100, 'record': 1,
     'retrain': False, 'rounds': 2, 'seed': 0, 'train_device': desktop-cpu-core-i7-7820x-fp32,
     'train_size': 50, 'verbose': 0,
}
|2024-10-19 17:03:48| ********************Experiment Start********************
|2024-10-19 17:03:57| Round=1 BestEpoch=  1 MAE=0.0012 RMSE=0.0016 NMAE=0.2504 NRMSE=0.3002 Training_time=0.2 s 
|2024-10-19 17:04:04| Round=2 BestEpoch=  1 MAE=0.0011 RMSE=0.0013 NMAE=0.2155 NRMSE=0.2568 Training_time=0.1 s 
|2024-10-19 17:04:04| ********************Experiment Results:********************
|2024-10-19 17:04:04| NMAE: 0.2330 ± 0.0174
|2024-10-19 17:04:04| NRMSE: 0.2785 ± 0.0217
|2024-10-19 17:04:04| MAE: 0.0011 ± 0.0001
|2024-10-19 17:04:04| RMSE: 0.0014 ± 0.0001
|2024-10-19 17:04:04| Acc_10: 0.2942 ± 0.0065
|2024-10-19 17:04:04| train_time: 0.1062 ± 0.0479
|2024-10-19 17:04:05| ********************Experiment Success********************
```

<div  align="center"> 
<img src="../fig/17_03_48_Model_ours_cpu_S50_R100_Ablation6.pdf" 
width = "900" height = "800" 
alt="1" align=center />
</div>
