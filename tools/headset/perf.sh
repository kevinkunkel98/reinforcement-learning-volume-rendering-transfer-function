#!/bin/zsh
# Samples the headset compositor's VRPerfMgr lines for N seconds; prints median FPS / GPU%.
secs=${1:-10}
adb logcat -c
sleep $secs
adb logcat -d | grep "VRPerfMgr.*FPS:" | sed -E 's/.*FPS: ([0-9]+).*GPU_Usage: ([0-9]+).*GPU_Bound: ([0-9]+).*/\1 \2 \3/' \
 | sort -n | awk '{f[NR]=$1; g[NR]=$2; b+=$3} END {if (NR==0) {print "no samples"; exit} m=int((NR+1)/2); print "samples=" NR " fps_median=" f[m] " fps_min=" f[1] " gpu_bound_share=" b/NR}'
adb logcat -d | grep "VRPerfMgr.*FPS:" | sed -E 's/.*GPU_Usage: ([0-9]+).*/\1/' | sort -n | awk '{g[NR]=$1} END {if (NR) print "gpu_median=" g[int((NR+1)/2)]}'
