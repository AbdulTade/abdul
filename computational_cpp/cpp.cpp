
#include <iostream>
#include <cmath>

using namespace std;

double differentiate(float );
double integrate(float ,float );

int main(){

short int option;

cout << "A program to determine the derivative and integral of exp(-x) with respect to x "<< endl;
cout << " Enter an option to differentiate or integrate 1. Integrate 2. Differentiate" << endl;
cin >> option;

switch(option){

    case 1:
        
         float lima,limb;
         cout << "Enter the lower limit: ";
         cin >> lima;
         cout << "\nEnter the upper limit: ";
         cin >> limb;

         double integral = integrate(lima,limb);
         cout << "Integral of exp(-x) is "<< integral <<endl;
         break;

    case 2:

         float val;
         cout << "Enter the point for which the derivative should be evaluated: ";
         cin >> val;

         double differential = differentiate(val);
         cout << "Derivative of exp(-x) is "<< differential <<endl;
        break;

    default:

          break;
}


return 0;

}

double differentiate(float val){

double interval = 0.00000001;
double val1 = exp(-interval);
double val2 = exp(-(val+interval));

double diff = (val2 - val1)/interval;

return diff;


}

double integrate(float lima,float limb){

long double Areasum = 0;
double width = (limb - lima)/1000000;

for(int i = 0; i < 1000000; i++){

double vali = lima + i * width;
Areasum += exp(-vali);

}

return Areasum;


}
