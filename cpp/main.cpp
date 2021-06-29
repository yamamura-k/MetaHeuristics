#include <iostream>
#include "ABC.cpp"

double sphere(ublas::vector<double> const x)
{
   return ublas::inner_prod(x, x);
}

int main()
{
   double opt = ABC::optimize<double>(2, 10, 5, 30, sphere);
   std::cout << opt << std::endl;
   return 0;
}