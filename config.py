_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/myseg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# ============================================================
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
num_classes = 7

custom_imports = dict(
    imports=[
        'semantic_segmentation.mmseg_mambavision_backbone',
        'semantic_segmentation.hooks.save_color_mask_hook',
    ],
    allow_failed_imports=False
)
# ============================================================
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# ============================================================
model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        type='MM_mamba_vision',
        out_indices=(0, 1, 2, 3),
        pretrained=None,

        depths=(1, 3, 8, 4),
        num_heads=(2, 4, 8, 16),
        window_size=(8, 8, 64, 32),
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.3,
        norm_layer='ln2d',  
        layer_scale=None,
    ),

    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=num_classes,
        norm_cfg=norm_cfg,
    ),

    auxiliary_head=dict(
        in_channels=256,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
    ),
)

# ============================================================
optim_wrapper = dict(
    _delete_=True,        
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }
    )
)

# ============================================================
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# ============================================================
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=10000,   
    val_interval=500    
)

val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
    classwise=True,   
)

# ============================================================
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,        
        save_best='mIoU',      
        rule='greater',
        max_keep_ckpts=3       
    )
)

custom_hooks = [
    dict(
        type='SaveColorMaskHook',
        save_dir='BVSeg_result',
        palette={
            0: (0, 0, 0),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (255, 255, 0),
            4: (0, 255, 255),
            5: (255, 0, 255),
            6: (128, 128, 128),
        }
    )
]

# ============================================================
test_dataloader = None
test_cfg = None
test_evaluator = None







