# StmTrt

## how to run
```
python create-onnx-models.py
cd trt
mdkir build
cd build
cmake ..
make 
```

## TODO list:
- [x] reduce cost of memcpy in fm of head

- [x] some cleaning code 

- [ ] add `#ifdef` for seprating x86 and aarch64

- [ ] int8 calibration (use regular funtion for calibriaction and make them generic)

- [x] work on `get_crop_single()` function

- [ ] use fp16 `__half` for data

- [ ] process box and score in cuda cores 

- [x] add code for saving engine and loading from engine file 

- [ ] solve problem of creating engine for head 

- [ ] Subnormal FP16 values detected

- [x] use pin memory for image [check this](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#h2d-d2h-data-trans-pci-band)

- [ ] use DEFINE for index of buffers in code 

- [ ] add limit for saving features in memory
