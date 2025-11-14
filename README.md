# Riva ASR Library - CUDA Feature Extractor
This is the Kaldi CUDA feature extractor. (Pulled out as a small, modular, standalone library.)

## Build
Docker Images
- You can use one of these [Kaldi containers from Nvidia NGC.](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/kaldi?version=23.11-py3)
- It has been tested with `nvcr.io/nvidia/kaldi:21.04-py3`

To build `libriva-asrlib-cuda-feature-extractor.a`:

```
# From the repository root...
image="nvcr.io/nvidia/kaldi:21.04-py3"
docker image pull $image

docker run -e NVIDIA_VISIBLE_DEVICES=0 --runtime=nvidia \
    --user $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    --cap-add=SYS_ADMIN --workdir=/opt/kaldi/  --rm -it -v `pwd`:/opt/kaldi/ \
    $image  /bin/bash

mkdir build
cd build
cmake ..
make -j

```

## Usage
Create the configuration object:
- `include/cu-feat-extr/options/online-batched-feature-info.h`
- Instantiate a `OnlineBatchedFeatureInfo`.
    - Use this overload `OnlineBatchedFeatureInfo(RivaASRLibFeat_SpectralFeatureType feature_type_in, bool use_cmvn_in)`
    - This is the only config object that should be directly instantiated. It will automatically construct all sub-config objects.
- Modify the config member variables as desired.

Create the compute pipeline object:
- `include/cu-feat-extr/cudafeat/online-batched-feature-pipeline-cuda.h`
- Instantiate a `OnlineBatchedFeaturePipelineCuda`
- Use `pipeline.GetMaxChunkFrames()` to help resize and prepare IO variables for the compute call below.
- Call `pipeline.ComputeFeaturesBatched(...)`
- Sync on `cudaStreamPerThread` as necessary.

## Notes
- The library efficiently supports any combination of (online, offline) X (Batch Size = 1, Batch Size > 1).
- The following will be deprecated in a future release:
    - The C API for the Cuda Matrix class: `c-cu-matrix.h`. Use `cu-matrix.h` directly.
    - This overload of the configuration constructor:
        - `OnlineBatchedFeatureInfo(const RivaASRLibFeat_OnlineBatchedFeatureInfo &opts_in)`
- Syntax for configuration objects may change in a trivial way in a future release.
- Symbols in the following files will be put into a cpp namespace in a future release:
    - `cu-container-helper-types.h`
    - `pipeline-helper-types.h`
- There is limited support and testing for compatibility with Librosa.
    - For configurations that use `frame_extraction_options.librosa_center = true`, future releases of the library may produce slightly different output (frame count and frame content) or have a different interface to manage the padding. These both apply especially for online mode.
    - The configuration variable `fbank_options.librosa_compat` may be split up into multiple fine grain variables in a future release.
- Forked from Kaldi around commit 7f57eaa08093ba148e3d2abdf6c337212c130214
