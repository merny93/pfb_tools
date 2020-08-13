//nvcc -o pfb_cuda pfb_cuda.cu -lgomp -lfftw3f -lm -lcufft -lcufftw
//nvcc -Xcompiler -fPIC -shared -o libpfb_cuda.so  pfb_cuda.cu -lcufft -lgomp to compile library


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <fftw3.h>
#include <omp.h>
#include <cufft.h>

#define NTHREAD_PFB 128

/*--------------------------------------------------------------------------------*/
struct PFB_GPU_PLAN {
  int n;
  int nchan;
  int nchunk;
  int ntap;
  int nthread;
  
  float *win_gpu;
  float *dat_gpu;
  float *dat_tapered_gpu;
  cufftComplex *dat_trans_gpu;
  float *pfb_gpu;
  cufftHandle cuplan;
};

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
struct PFB_GPU_PLAN *setup_pfb_plan(int n, int nchan, int ntap)
{
  struct PFB_GPU_PLAN *tmp=(struct PFB_GPU_PLAN *)malloc(sizeof(struct PFB_GPU_PLAN));
  int nn=nchan*ntap;
  float *win=(float *)malloc(sizeof(float)*nn);
  coswin(win,nn);
  mul_sinc(win,nn,ntap);

  int nchunk=get_nchunk(n,nchan,ntap);

  tmp->n=n;
  tmp->nchan=nchan;
  tmp->nchunk=nchunk;
  tmp->ntap=ntap;
  tmp->nthread=NTHREAD_PFB;
  if (cudaMalloc((void **)&(tmp->dat_gpu),sizeof(float)*n)!=cudaSuccess)
    printf("Malloc error on dat_gpu.\n");

  if (cudaMalloc((void **)&(tmp->dat_tapered_gpu),sizeof(float)*nchunk*nchan)!=cudaSuccess)
    printf("Malloc error on dat_tapered_gpu.\n");

  if (cudaMalloc((void **)&(tmp->win_gpu),sizeof(float)*nn)!=cudaSuccess)
    printf("Malloc error on win_gpu.\n");

  if (cudaMalloc((void **)&(tmp->dat_trans_gpu),sizeof(cufftComplex)*nchan*nchunk)!=cudaSuccess)
    printf("Malloc error on dat_trans_gpu.\n");
  if (cudaMalloc((void **)&(tmp->pfb_gpu),sizeof(float)*nchan)!=cudaSuccess)
    printf("Malloc error on pfb_gpu.\n");

  if (cudaMemcpy(tmp->win_gpu,win,nn*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    printf("Copy error on win_gpu.\n");


  if (cufftPlan1d(&(tmp->cuplan),nchan,CUFFT_R2C,nchunk)!=CUFFT_SUCCESS)
    printf("we had an issue creating plan.\n");

  
  return tmp;
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void setup_pfb_plan_wrapper(int n, int nchan, int ntap, struct PFB_GPU_PLAN **ptr)
//assumes **ptr already has been malloced, so ptr[0] will be the correct pointer when done
{
  //printf("n,nchan, and ntap are %d, %d, and %d\n",n,nchan,ntap);
  ptr[0]=setup_pfb_plan(n,nchan,ntap);
  //printf("plan address is %ld\n",(long)(ptr[0]));
  //printf("n is now %d\n",ptr[0]->n);
}
}
/*--------------------------------------------------------------------------------*/

void destroy_pfb_gpu_plan(struct PFB_GPU_PLAN *plan)
{
  cufftDestroy(plan->cuplan);
  cudaFree(plan->dat_gpu);
  cudaFree(plan->win_gpu);
  cudaFree(plan->dat_tapered_gpu);
  cudaFree(plan->dat_trans_gpu);
  cudaFree(plan->pfb_gpu);

  free(plan);
  
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void destroy_pfb_gpu_plan_wrapper(struct PFB_GPU_PLAN **plan)
{
  destroy_pfb_gpu_plan(plan[0]);
}
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void print_pfb_plan(struct PFB_GPU_PLAN *plan)
{
  printf("Printing PFB plan.\n");
  printf("N is %d\n",plan->n);
  printf("ntap is %d\n",plan->ntap);
  printf("nchan is %d\n",plan->nchan);
}
}
  
/*--------------------------------------------------------------------------------*/

void format_data(float *dat, int n, int nchan, int ntap, float *win, float **dat_out, int *nchunk)
{
  int nn=n/nchan-ntap;
  float *dd=(float *)malloc(sizeof(float)*nn*nchan);
  memset(dd,0,sizeof(float)*nn*nchan);
  for (int i=0;i<nn;i++)
    for (int j=0;j<ntap;j++)
      for (int k=0;k<nchan;k++)
	dd[i*nchan+k]+=dat[(i+j)*nchan+k]*win[j*nchan+k];
  *nchunk=nn;
  *dat_out=dd;
}

/*--------------------------------------------------------------------------------*/
__global__
void gpu_int162float32(short *in,float *out,int n)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  int nthread=gridDim.x*blockDim.x;
  for (int i=0;i<n;i+=nthread)
    if (myi+i<n)
      out[myi+i]=in[myi+i];
  
}
/*--------------------------------------------------------------------------------*/
__global__
void format_data_gpu(float *dat, int nchunk, int nchan, int ntap, float *win, float *dat_out)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  for (int i=0;i<nchunk;i++) {
    float tot=0;
    for (int j=0;j<ntap;j++) {
      tot+=dat[(i+j)*nchan+myi]*win[j*nchan+myi];
    }
    dat_out[i*nchan+myi]=tot;
  }
}
/*--------------------------------------------------------------------------------*/
__global__
void sum_pfb_gpu(cufftComplex *dat_trans, int nchan, int nchunk, float *pfb_out)
{
  int myi=blockIdx.x*blockDim.x+threadIdx.x;
  float tot=0;
  for (int i=0;i<nchunk;i++) {
    cufftComplex tmp=dat_trans[myi+i*nchan];
    tot+=tmp.x*tmp.x+tmp.y*tmp.y;
  }
  pfb_out[myi]=tot;
}
/*--------------------------------------------------------------------------------*/
void pfb_gpu(float *dat, float *pfb, struct PFB_GPU_PLAN *pfbplan)
{
  if (cudaMemcpy(pfbplan->dat_gpu,dat,pfbplan->n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess)
    printf("Copy error on dat_gpu.\n");
  for (int i=0;i<10;i++) {
    format_data_gpu<<<pfbplan->nchan/pfbplan->nthread,pfbplan->nthread>>>(pfbplan->dat_gpu,pfbplan->nchunk,pfbplan->nchan,pfbplan->ntap,pfbplan->win_gpu,pfbplan->dat_tapered_gpu);
    if (cufftExecR2C(pfbplan->cuplan, pfbplan->dat_tapered_gpu, pfbplan->dat_trans_gpu)!=CUFFT_SUCCESS)
      printf("Error executing FFT on GPU.\n");
    
    sum_pfb_gpu<<<pfbplan->nchan/pfbplan->nthread,pfbplan->nthread>>>(pfbplan->dat_trans_gpu,pfbplan->nchan,pfbplan->nchunk,pfbplan->pfb_gpu);
  }
  if (cudaMemcpy(pfb,pfbplan->pfb_gpu,sizeof(float)*pfbplan->nchan,cudaMemcpyDeviceToHost)!=cudaSuccess)
    printf("Error copying PFB to cpu.\n");
  
}


/*--------------------------------------------------------------------------------*/
void pfb_gpu16(short int *dat, float *pfb, struct PFB_GPU_PLAN *pfbplan)
{
  if (cudaMemcpy(pfbplan->dat_tapered_gpu,dat,pfbplan->n*sizeof(short int),cudaMemcpyHostToDevice)!=cudaSuccess)
    printf("Copy error on dat_gpu.\n");
  gpu_int162float32<<<8*pfbplan->nchan/pfbplan->nthread,pfbplan->nthread>>>((short int *)pfbplan->dat_tapered_gpu,pfbplan->dat_gpu,pfbplan->n);
  
  format_data_gpu<<<pfbplan->nchan/pfbplan->nthread,pfbplan->nthread>>>(pfbplan->dat_gpu,pfbplan->nchunk,pfbplan->nchan,pfbplan->ntap,pfbplan->win_gpu,pfbplan->dat_tapered_gpu);
  if (cufftExecR2C(pfbplan->cuplan, pfbplan->dat_tapered_gpu, pfbplan->dat_trans_gpu)!=CUFFT_SUCCESS)
    printf("Error executing FFT on GPU.\n");
  
  sum_pfb_gpu<<<pfbplan->nchan/pfbplan->nthread,pfbplan->nthread>>>(pfbplan->dat_trans_gpu,pfbplan->nchan,pfbplan->nchunk,pfbplan->pfb_gpu);
  
  if (cudaMemcpy(pfb,pfbplan->pfb_gpu,sizeof(float)*pfbplan->nchan,cudaMemcpyDeviceToHost)!=cudaSuccess)
    printf("Error copying PFB to cpu.\n");
  
}
/*--------------------------------------------------------------------------------*/
extern "C" {
void pfb_gpu16_wrapper(short int *dat, float *pfb, struct PFB_GPU_PLAN **pfbplan)
{
  pfb_gpu16(dat,pfb,pfbplan[0]);
}
}

/*================================================================================*/
#if 0

int main(int argc, char *argv[])
{
  long n;
  FILE *infile;
  infile=fopen("random_dat.raw","r");
  fread(&n,sizeof(long),1,infile);
  printf("N is %ld\n",n);

  float *dat=(float *)malloc(sizeof(float)*n);
  fread(dat,sizeof(float),n,infile);
  fclose(infile);
  printf("First element is %f\n",dat[0]);
  int nchan=3584*4;
  //int nchan=4096*4;
  int ntap=4;
  //int nn=nchan*ntap;
  int niter=1000;

  struct PFB_GPU_PLAN *pfbplan=setup_pfb_plan(n,nchan,ntap);
  float *pfb_sum=(float *)malloc(sizeof(float)*nchan);
  memset(pfb_sum,0,sizeof(float)*nchan);
  
#if 0
  short int *dd=(short int *)malloc(n*sizeof(short int));
#else
  short int *dd;
  if(cudaMallocHost(&dd,sizeof(short int)*n)!=cudaSuccess)
    printf("cuda malloc error on dd.\n");
#endif
  memset(dd,0,sizeof(short int)*n);

  for (int i=0;i<n;i++)
    dd[i]=1000*dat[i];
  
  double t1=omp_get_wtime();
  for (int i=0;i<niter;i++) {
    //pfb_gpu(dat,pfb_sum,pfbplan);// this is the float version
    pfb_gpu16(dd,pfb_sum,pfbplan);
  }
  double t2=omp_get_wtime();
  double throughput=1.0*nchan*pfbplan->nchunk*niter/(t2-t1)/1e6;
  printf("pfb[0] is now %12.4g, with time per iteration %12.4e and throughput %12.4f Msamp/s\n",pfb_sum[0],(t2-t1)/niter,throughput);

  float *tmpv=(float *)malloc(sizeof(float)*n);
  if (cudaMemcpy(tmpv,pfbplan->dat_gpu,n*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
    printf("Error copying temp data back to memory.\n");
  printf("vals are %12.4f %d\n",tmpv[0],dd[0]);
  destroy_pfb_gpu_plan(pfbplan);


  
#if 0  //this will do the pfb via fftw on the cpu
  int rank=1;
  //fftwf_complex *crap=fftwf_malloc(nchunk*(nchan/2+1)*sizeof(fftwf_complex));
  fftwf_complex *crap=fftwf_alloc_complex(nchunk*(nchan/2+1));
  //fftwf_plan plan=fftwf_plan_many_dft_r2c(rank,&nchan,nchunk,dat_out,NULL,1,nchan,crap,NULL,1,nchan/2+1,FFTW_ESTIMATE);
  fftwf_plan plan=fftwf_plan_many_dft_r2c(rank,&nchan,nchunk,dat_out,NULL,1,nchan,crap,NULL,1,nchan/2+1,FFTW_ESTIMATE);
  fftwf_execute(plan);
  outfile=fopen("out_dat.raw","w");
  fwrite(&nchan,1,sizeof(int),outfile);
  fwrite(&nchunk,1,sizeof(int),outfile);
  fwrite(dat_out,nchan*nchunk,sizeof(float),outfile);
  fclose(outfile);
  outfile=fopen("out_trans.raw","w");
  int asdf=nchan/2+1;
  fwrite(&asdf,1,sizeof(int),outfile);
  fwrite(&nchunk,1,sizeof(int),outfile);

  fwrite(crap,asdf*nchunk,sizeof(fftwf_complex),outfile);
  fclose(outfile);
  #endif
}
#endif
