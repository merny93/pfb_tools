import numpy as np
import ctypes
import time


mylib=ctypes.cdll.LoadLibrary("libpfb_cuda.so")
setup_pfb_plan_c=mylib.setup_pfb_plan_wrapper
setup_pfb_plan_c.argtyptes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]

destroy_pfb_gpu_plan_c=mylib.destroy_pfb_gpu_plan_wrapper
destroy_pfb_gpu_plan_c.argtypes=[ctypes.c_void_p]

print_pfb_plan_c=mylib.print_pfb_plan
print_pfb_plan_c.argtypes=[ctypes.c_void_p]

pfb_gpu16_c=mylib.pfb_gpu16_wrapper
pfb_gpu16_c.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]


def _setup_pfb_plan(n,nchan,ntap):
    plan=np.zeros(1,dtype='int64')    
    setup_pfb_plan_c(n,nchan,ntap,plan.ctypes.data)
    #print("plan address is ",plan[0])
    #print_pfb_plan_c(plan.ctypes.data)
    return plan
def _destroy_pfb_plan(plan):
    destroy_pfb_gpu_plan_c(plan.ctypes.data)


def pfb_gpu16(dat,nchan,ntap):
    #convenient wrapper if you don't want to think about setting up plans, etc.  you probably don't want to do this for real runs, though.
    n=len(dat)
    if dat.dtype!='int16':
        print('recasting data to int16')
        dat=np.asarray(dat,dtype='int16')
    plan=_setup_pfb_plan(n,nchan,ntap)
    ans=np.zeros(nchan,dtype='float32')
    pfb_gpu16_c(dat.ctypes.data,ans.ctypes.data,plan.ctypes.data)
    _destroy_pfb_plan(plan)
    return ans


nchan=4096*8
ntap=4
nset=1000
niter=100
x=np.random.randn(nset*nchan)
x=np.asarray(1000*x,dtype='int16')


plan=_setup_pfb_plan(nchan*nset,nchan,ntap)
ans=np.zeros(nchan,dtype='float32')

t1=time.time()
for i in range(niter):
    ans[:]=0
    pfb_gpu16_c(x.ctypes.data,ans.ctypes.data,plan.ctypes.data)
    #y=pfb_gpu16(x,nchan,ntap)
t2=time.time()
_destroy_pfb_plan(plan)
print('elapsed time was ',t2-t1,' with throughput ',niter*nset*nchan/(t2-t1)/1e6," Msamp/second")
