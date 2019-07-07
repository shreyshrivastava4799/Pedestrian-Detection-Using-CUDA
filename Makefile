NVCC = nvcc

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

convolution: headers.h kernel.cu host.cu  
	$(NVCC) -I  ./ host.cu kernel.cu  -o convolution $(LIBS) 
      
