# Usage
The output from step 3 will be a pointer to float. 
#### 1. Use `typecastHistograms(float *)` to convert and store the return int pointer.
#### 2. Pass the output to `step4(int *, size_t)` as first parameter. The second parameter can be calculated by:
```C++
int numElementsIn = 16*8*9;
size_t sizeIn = numElementsIn * sizeof(int);
```
Pass `sizeIn` as the second parameter to `step4()`      
#### 3. The output of `step4` (a float pointer) is the required feature vector of length 3780.