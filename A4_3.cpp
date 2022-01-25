// Assignement 4 - Alexander Railton - 250848086
# include <cmath>
# include <iostream>
# include <fstream>
# include <string>
# include <cassert>
# include <limits> 
# include <iomanip>
# include <iomanip>

void CRANK(double ** M, int n , int m, double lambda, double k, double * p , double * q, double aa, double bb, double cc, double a);

void print_file(const std::string & x, double ** A, int n, int m);

int main(int argc, const char * argv[])
{
    double lambda = 1.0;
    double a = 1.0;
    double h = .1;
    double k = lambda*h;
    int xleft = -1;
    int xright = 1;
    double TIME_MAX = 1.0;
    
    // New Parameters 
    double aa = (-1*a)*lambda/4;
    double bb = 1;
    double cc = -1 * aa;

    // nodes spacial and time from handout
    int m = std::floor((xright - xleft)/h)+1;
    int n = std::floor(TIME_MAX/k)+1;


    // Dyanmically allocate the arrays P and Q
    double * p = new double[m];
    double * q = new double[m];

    p [1] = 0;
    q [1] = 0;
    // Approximated Solution Intialization
    double ** M;
    M = new double *[n];
    M[0] = new double [n * m];

    // Allocate to adress for the indeces of pointer array
    for(int i=1; i<n; i++)
    {
        M[i] = &M[0][i * m];
    }
    
    // Initial Conditions
    for (int i = 0; i < m; i++)
    {
        if ((std::abs(xleft + (i*h))) <= 0.5) // initial condition is checked along spacial steps
        {
            M[0][i] = std::pow(std::cos(M_PI * ((xleft + (i * h)))),2);
        }
        else
        {
            M[0][i] = 0;
        }
        
    }
    
    // boundary Condition
    for(int t = 0; t < n; t++)
    {
        M[t][0] = 0;
    }

    //FT_BS(M, n , m , lambda);
    CRANK(M, n, m, lambda, k, p, q, aa, bb, cc, a);
    print_file("Assignement4aa.dat", M , n, m);

    
    // deallocate memory
    delete [] p;
    delete [] q;

    delete [] M[0];
    delete [] M;
    return 0;
}


// C) Lax-Friedrichs scheme 
void CRANK(double ** M, int n , int m, double lambda, double k, double * p , double * q, double aa, double bb, double cc, double a)
{
    // run FT-BS Scheme
    // loop throught to get next time step and apply BC
    for(int i = 1; i < n; i++)
    {
        for(int j = 1; j < m; j++)
        {
            double dd = M[i][j] - a * lambda * ((M[i][j+1] - M[i][j-1]) / 2);
            double denom = (aa * p[j] + bb);
            p[j+1] = -cc / denom;
            q[j+1] = (dd - q[j] * aa) / denom;
            // apply BC to last point
            if(j == m-1)
            {
                // Numerical BC A) - Method 1
                // M[i][m] = (1 - k) * M[i-1][j] - lambda * ( M[i][j] - M[i][j-1]);
                // Numerical BC B) - Method 2
                 M[i][j] = 2 * M[i][j-1] - M[i][j-2];
            }
            else
            {
                M[i][j] = p[j+1] * (k+1) * M[i-1][j] + (-1 * a *lambda / 4) * (M[i][j+1] - M[i][j-1] + M[i-1][j+1] - M[i-1][j-1]) + q[i+1];
            }
            
        }
    }
}

void print_file(const std::string & x, double ** M, int n, int m)
{
    std::ofstream write_file(x);
    assert(write_file.is_open()); // ensure file is open break

    write_file.setf(std::ios::scientific);
    write_file.setf(std::ios::showpos);
    write_file.precision(13);

    for(int i = 0; i < m; i++)
    {
        write_file << M[n - 1][i];
        if (i != (m-1))
        {
            write_file << std::endl;
        }
    }

    write_file.close();
}

