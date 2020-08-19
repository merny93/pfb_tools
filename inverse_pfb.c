#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <cblas.h>
#include "lapack.h"
//#include <clapack.h>


struct INVERSE_PFB_PLAN {
    int nchan;
    int nblock;
    int ntap;
    double *irfft_pnt;
    float *coeff_P;
    float *coeff_PPT;
    float *band_P;
    float *band_PPT;
    fftw_plan dft_plan;

};


//reusing the same window function as jon used

void coswin(float *vec, int n)//this is actually the hanning window function
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
//create the irfft plan execute it with fftw_execute(plan);
fftw_plan make_fftw_plan(fftw_complex *input_array, double *output_array, int nchan, int nblock){
    int rank = 1;
    int nlen = 2*(nchan -1);
    int istride = 1;
    int ostride = 1;
    fftw_plan plan = fftw_plan_many_dft_c2r(rank, &nlen, nblock, input_array, NULL, 1, nlen, output_array, NULL, 1, nchan, FFTW_ESTIMATE);
    return plan;
}
//lol that was useless
void free_fftw_plan(fftw_plan plan){
    fftw_destroy_plan(plan);
}
//note the the output of fft is effectivly already its own transpose in col major continous blocks


//returns pointer to the coeff_P matrix in column major form
float* make_coeff_P(int lblock, int ntap){
    int nn = lblock*ntap;
    float *P = (float*)malloc(nn*sizeof(float));
    coswin(P,nn);
    mul_sinc(P,nn,ntap);
    //now we need to reshape the array to be colum 
    float *PT = (float*)malloc(nn*sizeof(float));
    for (int i =0; i<nn; i++){
        int ncol = i/ntap;
        int nrow = i*lblock % nn;
        PT[i] = P[nrow + ncol];
    }
    free(P)
    return PT;
}

float* make_coeff_PPT(float *w, int ntap, int lblock){
    float *soln = (float*)malloc(ntap*lblock*sizeof(float));
    for (int k = 0; k<lblock; k++){
        for (int l = 0; l<ntap; l++){
            float sum = 0;
            for (int i=0; i<(ntap-l); i++){
                sum += w[i + k*ntap] * w[i+ l + k*ntap];
            }
            soln[l+ntap*k] = sum;
        }
    }
    return soln
}

struct INVERSE_PFB_PLAN *setup_inverse_plan_internal(int nchan, int nblock, int ntap){
    struct INVERSE_PFB_PLAN *plan_point = (struct INVERSE_PFB_PLAN*)malloc(sizeof(struct INVERSE_PFB_PLAN));
    // plan_point->dft_plan = make_fftw_plan(input_pointer, output_pointer, nchan, nblock);
    plan_point->coeff_P = make_coeff_P(lblock, ntap);
    plan_point->coeff_PPT = make_coeff_PPT(plan_point->coeff_P, ntap, lblock);
    plan_point->irfft_pnt = (double*)malloc(lblock*ntap*sizeof(double));

}

void free_inverse_plan_internal(struct INVERSE_PFB_PLAN *plan){
    free(plan->coeff_P);
    free(plan->coeff_PPT);
    // free(plan->dft_plan);

}

void inverse_pfb_internal(float *pfb, float *rec_ts, struct INVERSE_PFB_PLAN *plan){
    //start by doing the irfft
    fftw_plan irdft_plan = make_fftw_plan(pfb, plan->irfft_pnt,plan->nchan, plan->nblock);
    fftw_execute(irdft_plan);
    free_fftw_plan(plan);
    //now we have the psudo_ts (transposed in column major fortran style)

    //alocate the band matrix in fortran style
    plan_point->band_P = (float *)malloc(plan->ntsblock * plan->ntap * sizeof(float));
    plan_point->band_PPT = (float *)malloc(plan->nblock * plan->ntap * sizeof(float));

    float *ts_col;
    ts_col = (double*)malloc();//not sure of shape yet
    for (int i_off = 0, i_off <plan->lblock;i_off++ ){
        //populate band_P
        for(int col = 0; col < plan->ntsblock; col++){
            for (int row =0; row < plan-> ntap; row++ ){
                band_P[row + (col*plan->ntap)] = plan->coeff_P[plan->ntap - row - 1 + (i_off * plan->ntap) ];
            }
        }
        //similarly populate band_PPT
        for(int col = 0; col < plan->nblock; col++){
            for (int row =0; row < plan-> ntap; row++ ){
                band_PPT[row + (col*plan->ntap)] = plan->coeff_PPT[plan->ntap - row - 1 + (i_off * plan->ntap) ];
            }
        }
        //get the single row of the psudo ts
        for (int i = 0; i < plan->nblock; i++){
            ts_col[i] = plan->irfft_pnt[i_off + i*plan->lblock]; //stride 
        }

        LAPACKE_dpbsv(LAPACK_COL_MAJOR, 'L', plan->nblock, ,1)
        ////PUT THE lapackpbsv call here o

        //put the dgbmv call next

        //this will return the column major rec_ts at row i_off which is already transposed when givin it back to numpy so is done
    
    }



}

extern "C" {
void setup_inverse_plan(struct INVERSE_PFB_PLAN **plan_ptr, int nchan, int nblock, int ntap){
    *plan_ptr = setup_inverse_plan_internal();

}
}

extern "C" {
void free_inverse_plan(struct INVERSE_PFB_PLAN *plan_ptr){
    free_inverse_plan_internal(plan_ptr);

}
}

extern "C" {
void inverse_pfb_cwrap(float *pfb, float *rec_ts, struct INVERSE_PFB_PLAN **plan){
    inverse_pfb_internal(pfb,rec_ts,*plan);
}
}
//to create the coeff_P:
/*
int nn=lblock*ntap;
float *win=(float *)malloc(sizeof(float)*nn);
coswin(win,nn);
mul_sinc(win,nn,ntap);
*/
//and then recall that a stride across vertical dimension is lblock which is 2*(nchan-1) <-------

//nblock never changed... its just the amount of blocks we are working on


//to create the coeff_PPT
//starting with coeff_P as P
/*
def coeff_PPT(w):
    ntap = w.shape[0]
    nblock = w.shape[1]
    soln = np.zeros(ntap*nblock).reshape(ntap,nblock)
    for l in range(ntap):
        for k in range(nblock):
            sum = 0
            for i in range(ntap - l):
                sum +=w[i,k] * w[i+l,k]
            soln[l,k] = sum
    return soln
*/
// now we have the inverse plan

//time to iterate over the l block 
//and pass it to the  

//steps of inverse:
/*
-inverse of the pfb
-make the P matrix:
    - generate the coeficinets
    -stick it in the band matrix
-make the PPT matrix:
    -generate the coeficients 
    -stick them in the band matrix
-solve the ppt matrix 
-project into timestream basis

*/




void main(){
    return;
}