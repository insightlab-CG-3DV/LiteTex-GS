## DeepBlending scenes use this parameter set.
python train.py \
    -s ../data/deepblending/playroom \
    -m ../outputs/db/playroom \
    --iterations 30000 \
    --resolution_mode freq \
    --densify_mode freq \
    --densify_until_iter 15000 \
    --max_reso_scale 4 \
    --start_significance_factor 3 \
    --texture_warmup_iters 1000 \
    --densification_interval 300 \
    --adaptive_texelsize_percentile 0.85 \
    --lambda_l2 0.03 \
    --texture_regul_max_texels 2000000 \
    --eval \
    --log_psnr \
    --prune_opacity_threshold 0.015 \
    --prune_opacity_threshold_final 0.04 \
    --prune_post_split_opacity_floor 0.01 \
    --prune_contrib_threshold 5e-6 \
    --prune_area_threshold 5e-5 \
    --prune_low_contrib_patience 8 \
    --prune_hard_interval 800 \
    --splitting_threshold 16

## MipNeRF360 scenes use this parameter set.
python train.py \
    -s ../data/mipNerf360/garden  \
    -m ../outputs/mipNerf360/garden \
    --iterations 30000 \
    --resolution_mode freq \
    --densify_mode freq \
    --densify_until_iter 15000 \
    --max_reso_scale 4 \
    --start_significance_factor 3 \
    --texture_warmup_iters 1000 \
    --densification_interval 300 \
    --adaptive_texelsize_percentile 0.85 \
    --lambda_l2 0.03 \
    --texture_regul_max_texels 2000000 \
    --eval \
    --log_psnr \
    --prune_opacity_threshold 0.015 \
    --prune_opacity_threshold_final 0.04 \
    --prune_post_split_opacity_floor 0.01 \
    --prune_contrib_threshold 5e-6 \
    --prune_area_threshold 5e-5 \
    --prune_low_contrib_patience 8 \
    --prune_hard_interval 800 \
    --splitting_threshold 16 

## Tanks and Temples scenes use this parameter set.
python train.py \
    -s ../data/tanksandtemples/truck \
    -m ../outputs/tt/truck \
    --iterations 30000 \
    --resolution_mode freq \
    --densify_mode freq \
    --densify_until_iter 15000 \
    --max_reso_scale 4 \
    --start_significance_factor 3 \
    --texture_warmup_iters 1000 \
    --densification_interval 300 \
    --adaptive_texelsize_percentile 0.6 \
    --lambda_l2 0.03 \
    --texture_regul_max_texels 2000000 \
    --eval \
    --log_psnr \
    --prune_opacity_threshold 0.015 \
    --prune_opacity_threshold_final 0.04 \
    --prune_post_split_opacity_floor 0.01 \
    --prune_contrib_threshold 5e-6 \
    --prune_area_threshold 5e-5 \
    --prune_low_contrib_patience 8 \
    --prune_hard_interval 800 \
    --splitting_threshold 16 \
    --cap_max 220000 

## Evaluation commands for the configured scenes.
python metrics.py -m ../outputs/db/playroom
python metrics.py -m ../outputs/mipNerf360/garden
python metrics.py -m ../outputs/tt/truck
