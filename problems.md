1. Error encountered: failed call to cuInit: CUDA_ERROR_UNKNOWN: unknown error
Solution: 
1. conda create -n tf-gpu tensorflow-gpu 
2. sudo modprobe -r nvidia_uvm 
3. sudo modprobe nvidia_uvm
4. sudo systemctl enable nvidia-suspend.service
5. sudo systemctl enable nvidia-resume.service
6. sudo systemctl enable nvidia-hibernate.service