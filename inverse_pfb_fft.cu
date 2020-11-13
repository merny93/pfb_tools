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


    //malloc for window ts which is of size nblocks by 2*(nchan-1)
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

    //malloc for window_ts_ft this one is nblocks/2 +1 by 2*(nchan-1)              
    if (cudaMalloc((void **)&(tmp->win_gpu)), sizeof((float)*(nblocks/2 +1)*2*(nchan-1))) != cudaSuccess){
        //same as dat_gpu 
        printf("Malloc error for window ft on gpu \n");
    }

    //do the fft on gpu of the window
    cufftHandle *win_plan;

    if (cufftPlanMany(win_plan, 1, &nblocks, &69, sizeof(float)*2*(nchan-1), sizeof(float)*1, &69, sizeof(float)*(nblocks/2 +1)*2*(nchan-1), sizeof(float)*1, CUFFT_R2C, 2*(nchan-1))  != CUFFT_SUCCESS){
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
    int psudo_ts_width = 2*(nchan-1);
    int psudo_ts_height = nblocks;
    if (cudaMalloc((void **)&(tmp->psudo_ts_gpu)), sizeof((float)*psudo_ts_width*psudo_ts_height) != cudaSuccess){
        //will need to be half the byte size of the  dat_gpu as its just reals now 
        printf("Malloc error for psudo ts on gpu \n");
    }

    //malloc for ft of psudo ts on gpu
    int psudo_ft_width = psudo_ts_width
    int psudo_ft_height = (int) (nblocks/2 +1)
    if (cudaMalloc((void **)&(tmp->psudo_ts_ft_gpu)), sizeof((float)*psudo_ft_height*psudo_ts_width)) != cudaSuccess){
        //same as dat_gpu 
        printf("Malloc error for psudo_ts_ft on gpu \n");
    } 


    // now time for the hard part 
    // starting with the fft to creat the psudo ts
    if (cufftPlanMany(&(tmp->psudo_ts_plan), 1, &psudo_ts_width, &69, sizeof(cufftComplex)*1, sizeof(cufftComplex)*nchan, &69, sizeof(float)*1, sizeof(float)*psudo_ts_width, CUFFT_C2R, nblocks)  != CUFFT_SUCCESS){
        printf("Failed to create the fft plan for psudo ts \n");
    }

    //now for the psudo ts ft from the previous size to the size of window and then we will go back 
    //stride it vertically so width between consecutive and 1 between sets
    //save "normally" so spacing between consecutive is 1 and between sets is the length of the transform
    if (cufftPlanMany(&(tmp->psudo_ft_plan), 1, &psudo_ts_height, &69, sizeof(float)*psudo_ts_width, sizeof(float)*1, &69, sizeof(cufftComplex)*1, sizeof(cufftComplex)*psudo_ft_height, CUFFT_R2C, nblocks)  != CUFFT_SUCCESS){
        printf("Failed to create the fft plan for psudo ft \n");
    }

    ///do the convolution here::: but this is the plan so no need yet:

    //now do the transform back
    //if read sequentially should be the data in order:
    if (cufftPlanMany(&(tmp->result_ift_plan), 1, &psudo_ts_height, &69, sizeof(cufftComplex)*1, sizeof(cufftComplex)*psudo_ft_height, &69, sizeof(float)*psudo_ts_width, sizeof(float)*1, CUFFT_C2R, nblocks)  != CUFFT_SUCCESS){
        printf("Failed to create the fft plan for rts \n");
    }

    return tmp
}
void inverse_pfb_gpu_internal(float *data, float *result, struct INVERSE_GPU_PLAN *plan){
    //copy data onto gpu
    if (cudaMemcpy(plan->dat_gpu,data,sizeof(cufftComplex)* (plan->nchan) * (plan->nblocks),cudaMemcpyHostToDevice)!=cudaSuccess){
        printf("data to gpu copy failed ");
    }

    //execute the first thing 
    if (cufftExecC2R(plan->psudo_ts_plan,plan->dat_gpu, plan->psudo_ts_gpu) != CUFFT_SUCCESS){
        printf("failed to do fft for psudo ts\n");
    }

    //now the second
    if (cufftExecR2C(plan->psudo_ft_plan, plan->psudo_ts_gpu, plan->psudo_ts_ft_gpu) != CUFFT_SUCCESS){
        printf("failed to do fft for psudo ts ft\n");
    }
    //do the convolution 
    convolve_pfb_gpu<<< something >>> sinetgugb else
    //do the filtering 
    filter_pfb_gpu<<< >>>>
    //and fft back 
    if (cufftExecC2R(plan->psudo_ft_plan, plan->psudo_ts_ft_gpu, plan->psudo_ts_gpu) != CUFFT_SUCCESS){
        printf("failed to do fft for psudo ts ft\n");
    }
    //i should really clean this up rip
    if (cudaMemcpy(result,plan->psudo_ts_gpu,sizeof(float)*2*((plan->nchan)-1)*(plan->nblocks),cudaMemcpyDeviceToHost)!=cudaSuccess){
        printf("Error copying RTS to cpu.\n");
    }
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