NVCC = nvcc

histogram: headers.h kernel2.cu host2.cu  
	$(NVCC) -I ./ kernel2.cu host2.cu -o histogram

run:
	./histogram

clean:
	rm -rf histogram

