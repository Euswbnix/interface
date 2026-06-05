# config.py

class InterfaceConfig:

    first_slow_model_configs = []
    first_mae_model_configs = [
        ('/home/smartlink_temp/workspace/data_4/best_model.pth',
         'MAE'),
    ]

    # second_slow_model_configs = []
    # second_mae_model_configs = [
    #     ('/video/data_3/checkpoints/best_model.pth',
    #      'MAE'),
    # ]


    second_slow_model_configs = [
        ('/home/smartlink_temp/workspace/work_dirs/slowfast_r50_only/slowfast_r50_only.py',
         '/home/smartlink_temp/workspace/work_dirs/slowfast_r50_only/best_acc_top1_epoch_27.pth',
         'SlowFast'),
    ]

    second_mae_model_configs = [
        ('/home/smartlink_temp/workspace/data_3/best_model.pth',
         'MAE'),
    ]

    device = 'cuda:0'
    clip_len = 64
    overlap = 20

