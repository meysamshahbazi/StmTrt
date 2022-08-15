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

- [ ] solve problem of creating engine for head 

- [ ] Subnormal FP16 values detected

- [ ] use pin memory for image ?

- [ ] use DEFINE for index of buffers in code 