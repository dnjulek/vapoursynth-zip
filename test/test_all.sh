#!/bin/sh

set -e

vspipe -p ./adaptive_binarize.vpy .
vspipe -p ./bilateral.vpy .
vspipe -p ./boxblur.vpy .
vspipe -p ./checkmate.vpy .
vspipe -p ./clahe.vpy .
vspipe -p ./color_map.vpy .
vspipe -p ./comb_mask.vpy .
vspipe -p ./comb_mask_mt.vpy .
vspipe -p ./limit_filter.vpy .
vspipe -p ./limiter.vpy .
vspipe -p ./packrgb.vpy .
vspipe -p ./plane_props.vpy .
vspipe -p ./planeminmax.vpy .
vspipe -p ./rfs_mismatch.vpy .
vspipe -p ./ssimulacra2.vpy .