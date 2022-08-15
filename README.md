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

- [x] work on `get_crop_single()` finction

- [ ] solve problem of creating engine for head 

- [ ] Subnormal FP16 values detected

- [x] use pin memory for image [check this](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#h2d-d2h-data-trans-pci-band)

- [ ] use DEFINE for index of buffers in code 

- [ ] add limit for saving features in memory
