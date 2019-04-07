NVCC = nvcc

step4: kernel4.cu host.cu
	$(NVCC) -I ./ kernel4.cu host.cu -o step4

clean:
	rm -rf step*


