#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <fftw3.h>
#include <omp.h>
#include <cufft.h>

/*
1 do the c-r inverse fft in the expected direciton
2 do the r-c fft in repeated time chungs
3 generate the window 
4 r-c fft window matrix
5 filter
6 deconvolve
7 c-r ifft and return
*/

/*--------------------------------------------------------------------------------*/

int get_nchunk(int n,int nchan,int ntap)
{
  return n/nchan-ntap;
}

/*--------------------------------------------------------------------------------*/

void coswin(float *vec, int n)
{
  for (int i=0;i<n;i++) {
    float xx=2.0*(i-n/2)/(n-1);
    vec[i]=0.5+0.5*cos(xx*M_PI);
  }
}

/*--------------------------------------------------------------------------------*/
void mul_sinc(float *vec, int n, int ntap)
{
  for (int i=0;i<n;i++) {
    float xx=ntap*1.0*(i-n/2)/(n-1);
    if (xx!=0)
      vec[i]=vec[i]*sin(M_PI*xx)/(M_PI*xx);    
  }
}
/*--------------------------------------------------------------------------------*/


struct INVERSE_GPU_PLAN {
    int nblocks;
    int nchan;
    int nchunk;
    int ntap;
    int nthread;
    
    float *win_gpu;
    cufftComplex *dat_gpu;
    float *psudo_ts_gpu;
    cufftComplex *psudo_ts_ft_gpu;
    float *dat_tapered_gpu;
    cufftComplex *dat_trans_gpu;
    float *pfb_gpu;
    cufftHandle psudo_ts_plan;
    cufftHandle psudo_ft_plan;
    cufftHandle result_ift_plan;
};


struct INVERSE_GPU_PLAN *setup_inverse_plan_internal(int nblocks, int nchan, int ntap){
    struct INVERSE_GPU_PLAN *tmp = (struct INVERSE_GPU_PLAN *)malloc(sizeof(struct INVERSE_GPU_PLAN));
    
    tmp->nblocks = nblocks;
    tmp->nchan = nchan;
    tmp->nchunk=nchunk;
    tmp->ntap=ntap;


    // starting with the window stuff which is persistant throughout:
    int win_size = ntap*2*(nchan-1);
    //its sparse so this is gonna be slow but it doesnt reallllly matter
    float *win_tmp = (float *)calloc(sizeof((float)*nblocks*2*(nchan-1)));
    coswin(win_tmp, win_size);
    mul_sinc(win_tmp, win_size);


    //malloc for window ts
    float *win_temp_gpu;
    if (cudaMalloc((void **)&win_temp_gpu), sizeof((float)*nblocks*2*(nchan-1)) != cudaSuccess){
        //will need to be half the byte size of the  dat_gpu as its just reals now 
        printf("Malloc error for window ts on gpu \n");
    }
    //copy window to window ts
    if (cudaMemcpy(tmp->win_gpu, win, nblocks*2*(nchan-1), cudaMemcpyHostToDevice)!= cudaSuccess){
        printf("Error copying window to gpu\n");
    }
    //no longer need the ram copy of window so free it
    free(win_tmp);

    //malloc for window_ts_ft
    if (cudaMalloc((void **)&(tmp->win_gpu)), sizeof((float)*(nblocks/2 +1)*2*(nchan-1))) != cudaSuccess){
        //same as dat_gpu 
        printf("Malloc error for window ft on gpu \n");
    }

    //do the fft on gpu of the window
    cufftHandle *win_plan;

    if (cufftPlanMany(win_plan, 1, &nblocks, &69, sizeof(float)*2*(nchan-1), sizeof(float)*1, &69, sizeof(float)*(nblocks/2 +1)*2*(nchan-1), sizeof(float)*1, CUFFT_R2C)  != CUFFT_SUCCESS){
        printf("Failed to create the fft plan for window \n");
    }

    if (cufftExecR2C(*win_plan,win_temp_gpu,tmp->win_gpu) != CUFFT_SUCCESS){
        printf("failed to do fft of window data\n");
    }

    //all done so free the plan
    cufftDestroy(*win_plan);
    //and the memory
    cudaFree(win_temp_gpu);


    //malloc the data on gpu
    if (cudaMalloc((void **)&(tmp->dat_gpu)), sizeof((float)*nblocks*nchan) != cudaSuccess){
        //might need to change that to double (64) or longdouble (128) cause its a complex array 
        //just use complex32 in numpy and this will work
        printf("Malloc error for incoming data on gpu \n");
    } 

    //malloc for psudo ts on gpu
    if (cudaMalloc((void **)&(tmp->psudo_ts_gpu)), sizeof((float)*nblocks*2*(nchan-1)) != cudaSuccess){
        //will need to be half the byte size of the  dat_gpu as its just reals now 
        printf("Malloc error for psudo ts on gpu \n");
    }

    //malloc for ft of psudo ts on gpu
    if (cudaMalloc((void **)&(tmp->psudo_ts_ft_gpu)), sizeof((float)*(nblocks/2 +1)*2*(nchan-1))) != cudaSuccess){
        //same as dat_gpu 
        printf("Malloc error for psudo_ts_ft on gpu \n");
    } 


    // now time for the hard part 
    if (cufftPlanMany(&(tmp->psudo_ts_plan),1, &))
    return tmp
}

extern "C"{
    void make_inverse_plan(int nblocks, int nchan, int ntap, struct INVERSE_GPU_PLAN **ptr){
        
        //allocate the pointer in advance!!!!
        ptr[0] = setup_inverse_plan_internal(nblocks, nchan, ntap);
    }
}



extern "C"{
    void inverse_pfb_gpu(float *data, float *result, struct INVERSE_GPU_PLAN **plan_point){
        //the double pointer bulshit is cause python only knows how to c type
        inverse_pfb_gpu_internal(data, result, plan_point[0]);
    }
}