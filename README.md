# trial_mmdetection

## memo
CUDA利用でエラー。ホストCUDAドライバとコンテナCUDAバージョンの違いが影響している可能性あり。
CUDAを利用する場合は、ホストCUDAドライバのダウングレード検討。

[CUDA Toolkit 12.1 のダウンロード](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)