import numpy as np
import scipy.linalg as la
import time

def cg_rc ( n, b, x, r, z, p, q, job, iterate, rho, rho_old, rlbl ):

#*****************************************************************************80
#
## CG_RC is a reverse communication conjugate gradient routine.
#
#  Discussion:
#
#    This routine seeks a solution of the linear system A*x=b
#    where b is a given right hand side vector, A is an n by n
#    symmetric positive definite matrix, and x is an unknown vector
#    to be determined.
#
#    Under the assumptions that the matrix A is large and sparse,
#    the conjugate gradient method may provide a solution when
#    a direct approach would be impractical because of excessive
#    requirements of storage or even of time.
#
#    The conjugate gradient method presented here does not require the
#    user to store the matrix A in a particular way.  Instead, it only
#    supposes that the user has a way of calculating
#      y = alpha * A * x + b * y
#    and of solving the preconditioned linear system
#      M * x = b
#    where M is some preconditioning matrix, which might be merely
#    the identity matrix, or a diagonal matrix containing the
#    diagonal entries of A.
#
#    This routine was extracted from the "templates" package.
#    There, it was not intended for direct access by a user
#    instead, a higher routine called "cg()" was called once by
#    the user.  The cg() routine then made repeated calls to
#    cgrevcom() before returning the result to the user.
#
#    The reverse communication feature of cgrevcom() makes it, by itself,
#    a very powerful function.  It allows the user to handle issues of
#    storage and implementation that would otherwise have to be
#    mediated in a fixed way by the function argument list.  Therefore,
#    this version of cgrecom() has been extracted from the templates
#    library and documented as a stand-alone procedure.
#
#    The user sets the value of JOB to 1 before the first call,
#    indicating the beginning of the computation, and to the value of
#    2 thereafter, indicating a continuation call.
#    The output value of JOB is set by cgrevcom(), which
#    will return with an output value of JOB that requests a particular
#    new action from the user.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    12 January 2013
#
#  Author:
#
#    John Burkardt
#
#  Reference:
#
#    Richard Barrett, Michael Berry, Tony Chan, James Demmel,
#    June Donato, Jack Dongarra, Victor Eijkhout, Roidan Pozo,
#    Charles Romine, Henk van der Vorst,
#    Templates for the Solution of Linear Systems:
#    Building Blocks for Iterative Methods,
#    SIAM, 1994,
#    ISBN: 0898714710,
#    LC: QA297.8.T45.
#
#  Parameters:
#
#    Input, integer N, the dimension of the matrix.
#
#    Input, real B(N), the right hand side vector.
#
#    Input, real X(N).  On first call, the user
#    should store an initial guess for the solution in X.
#
#    Input, real R(N), Z(N), P(N), Q(N), work arrays.  The user should
#    create each of these before the first call, using the zeros() command.
#    On subsequent calls, the user may be asked to assign a value to one
#    of these vectors.
#
#    Input, integer JOB, communicates the task to be done.
#    The user needs to set the input value of JOB to 1, before the first call,
#    and then to 2 for every subsequent call for the given problem.
#
#    Output, real X(N), the current solution estimate.
#    Each time JOB is returned as 4, X has been updated.
#
#    Output, real R(N), Z(N), P(N), Q(N), work arrays.  Depending on the
#    output value of JOB, the user may be asked to carry out a computation
#    involving some of these vectors.
#
#    Output, integer JOB, communicates the task to be done.
#    * JOB = 1, compute Q = A * P
#    * JOB = 2: solve M*Z=R, where M is the preconditioning matrix
#    * JOB = 3: compute R = R - A * X
#    * JOB = 4: check the residual R for convergence.  
#               If satisfactory, terminate the iteration.
#               If too many iterations were taken, terminate the iteration.
#
  import numpy as np
#
#  Initialization.
#  Ask the user to compute the initial residual.
#
  if ( job == 1 ):

    r = b.copy ( )

    job = 3
    rlbl = 2
#
#  Begin first conjugate gradient loop.
#  Ask the user for a preconditioner solve.
#
  elif ( rlbl == 2 ):

    iterate = 1

    job = 2
    rlbl = 3
#
#  Compute the direction.
#  Ask the user to compute ALPHA.
#  Save A*P to Q.
#
  elif ( rlbl == 3 ):

    rho = np.dot ( r, z )

    if ( 1 < iterate ):
      beta = rho / rho_old
      z = z + beta * p

    p = z.copy ( )

    job = 1
    rlbl = 4
#
#  Compute current solution vector.
#  Ask the user to check the stopping criterion.
#
  elif ( rlbl == 4 ):

    pdotq = np.dot ( p, q )
    alpha = rho / pdotq
    x = x + alpha * p
    r = r - alpha * q

    job = 4
    rlbl = 5
#
#  Begin the next step.
#  Ask for a preconditioner solve.
#
  elif ( rlbl == 5 ):

    rho_old = rho
    iterate = iterate + 1

    job = 2
    rlbl = 3

  return x, r, z, p, q, job, iterate, rho, rho_old, rlbl


def job1(A, P, lower):
    #Q = A*P where A is the input array and p is smthing the function gives to us 
    lda = A.shape[0]
    N = A.shape[1]
    k = lda -1
    Q = la.blas.dsbmv(k,1,A,P, lower=1)
    return Q

def job2(A,R):
    #solve M*Z=R, where M is the preconditioning matrix
    ##the precondition matrix is gonna be the diagonal entries of A
    Z = R / A[0,:]
    return Z

def job3(A, R, X, lower):
    #compute R = R - A * X
    lda = A.shape[0]
    N = A.shape[1]
    k = lda - 1
    R = R - la.blas.dsbmv(k,1,A,X, lower=1)
    return R

def job4(R):
    #check the residual R for convergence.
    res = la.norm(R)
    return res


def solve_banded_CG(band, vec, niter=False , tol=False ,lower = False):
    '''
        Magic conjuget gradient solver for sparce matricies using fast blas functions :)
        will work inside of the richard code
        takes the band matrix A 
        And a vec b
        and solves for x in Ax = b
    '''
    if lower is True:
        lower = 1
    elif lower is False:
        lower = 0
    else:
        return "get recked"
    job =1 
    rlbl = 0
    rho = 1
    rho_old = 1
    iterate = 0
    N = band.shape[1]

    if niter is False:
        niter = N

    R = np.zeros(N)
    Z = np.zeros(N)
    P = np.zeros(N)
    Q = np.zeros(N)
    X = np.zeros(N)
    X, R, Z, P, Q, job, iterate, rho, rho_old, rlbl = cg_rc(N, vec, X, R,Z,P,Q,job, iterate, rho, rho_old, rlbl)
    if job==3:
        R = job3(band, R, X, lower)
        job = 2
    else:
        print("crappp abort")
        return "fail"
    while iterate < niter:
        X, R, Z, P, Q, job, iterate, rho, rho_old, rlbl = cg_rc(N, vec, X, R,Z,P,Q,job, iterate, rho, rho_old, rlbl)
        if job == 1:
            Q = job1(band, P, lower)
        elif job ==2:
            Z = job2(band, R)
        elif job == 3:
            R = job3(band, R, X, lower)
        elif job ==4:
            if tol is False:
                pass
            else:
                res = job4(R)
                if res < tol:
                    return X
        job = 2
    return X

if __name__ == "__main__":
    ab = np.array([[ 4,  5,  6,  7, 8, 9],
                [ 2,  2,  2,  2, 2, 0],
               [-1, -1, -1, -1, 0, 0]])
    b = np.array([1, 2, 2, 3, 3, 3])
    t1 = time.time()
    for i in range(100):
        x = solve_banded_CG(ab, b,niter=6, lower=True)
    t2 = time.time()
    for i in range(100):
        y =  la.solveh_banded(ab, b, lower=True)
    t3 = time.time()
    print(la.norm(x-y))
    print("CG solve took", t2-t1, ".Classic solve took", t3-t2)

