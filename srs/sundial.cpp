#include <iostream>
#include <sundials/sundials_types.h>
#include <cvode/cvode.h>             // prototypes for CVODE functions and constants
#include <nvector/nvector_serial.h>  // serial N_Vector types, functions, and macros
#include <sunlinsol/sunlinsol_dense.h> // prototypes for dense linear solver
#include <sundials/sundials_math.h>  // contains macros ABS, SUNSQR, EXP

// User-supplied function called by the solver
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
    realtype y1 = NV_Ith_S(y,0);
    realtype y2 = NV_Ith_S(y,1);
    NV_Ith_S(ydot,0) = y2;
    NV_Ith_S(ydot,1) = -y1;
    return 0;
}

int main() {
    realtype reltol, t, tout;
    N_Vector y;
    void *cvode_mem;
    SUNMatrix A;
    SUNLinearSolver LS;
    int flag;

    // Create serial vector of length 2 for I.C.
    y = N_VNew_Serial(2);
    if (y == nullptr) {
        std::cerr << "Failed to create vector y." << std::endl;
        return 1;
    }
    NV_Ith_S(y,0) = 1.0; // y1
    NV_Ith_S(y,1) = 0.0; // y2

    // Call CVodeCreate to create the solver memory and specify the
    // Backward Differentiation Formula and the use of a Newton iteration
    cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    if (cvode_mem == nullptr) {
        std::cerr << "Failed to create CVODE solver memory." << std::endl;
        N_VDestroy(y);
        return 1;
    }

    // Call CVodeInit to initialize the integrator memory and specify the
    // user's right hand side function in y'=f(t,y), the initial time T0, and
    // the initial dependent variable vector y.
    flag = CVodeInit(cvode_mem, f, 0.0, y);
    if (flag != CV_SUCCESS) {
        std::cerr << "Failed to initialize CVODE." << std::endl;
        N_VDestroy(y);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Call CVodeSStolerances to specify the scalar relative tolerance
    // and the scalar absolute tolerance
    reltol = 1e-4;
    flag = CVodeSStolerances(cvode_mem, reltol, 1e-8);
    if (flag != CV_SUCCESS) {
        std::cerr << "Failed to set tolerances." << std::endl;
        N_VDestroy(y);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Create dense SUNMatrix for use in linear solves
    A = SUNDenseMatrix(2, 2);
    if (A == nullptr) {
        std::cerr << "Failed to create SUNMatrix." << std::endl;
        N_VDestroy(y);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Create dense SUNLinearSolver object for use by CVODE
    LS = SUNLinSol_Dense(y, A);
    if (LS == nullptr) {
        std::cerr << "Failed to create SUNLinearSolver." << std::endl;
        N_VDestroy(y);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Call CVDlsSetLinearSolver to attach the matrix and linear solver to CVODE
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (flag != CV_SUCCESS) {
        std::cerr << "Failed to set linear solver." << std::endl;
        N_VDestroy(y);
        SUNLinSolFree(LS);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Integrate over 10 units of time
    tout = 10.0;
    flag = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
    if (flag != CV_SUCCESS) {
        std::cerr << "Failed to integrate." << std::endl;
        N_VDestroy(y);
        SUNLinSolFree(LS);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        return 1;
    }

    // Print final solution
    std::cout << "At t = " << t << ", y = " << NV_Ith_S(y,0) << ", " << NV_Ith_S(y,1) << std::endl;

    // Free y vector
    N_VDestroy(y);

    // Free integrator memory
    CVodeFree(&cvode_mem);

    // Free linear solver and matrix
    SUNLinSolFree(LS);
    SUNMatDestroy(A);

    return 0;
}
