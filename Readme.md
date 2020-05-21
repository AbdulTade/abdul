import math
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt
import unicodedata

importing necessary modules to be used in the program.


exp = math.exp
log = math.log10
cos = math.cos
sin = math.sin
tan = math.tan .....

Creating references to important math functions. References are only executed when parenthesis is added to reference name.

round(num)
params: number to be rounded.
Type: function
Round function is to enable rounding of numbers to whole numbers. returns integer but sometimes float due to the in accuracies of the math.floor function used in the code

ndecimalPlaces(num,n)
params: number to be round (num) and the number of decimal places it should be rounded to (n)
Type: function
Rounds the parameter to n decimal places.Returns number with n decimal Places
eg. ndecimalPlaces(13.46457,2) will yield 13.46

negativeDetector(num)
params: integer number
Type: function
formats text number so as to make sign distinct from value.Returns a string
eg. negativeDetector(-5) will return the string ' - 5'

differentiate
Type: class
Takes a string of the function to be differentiated and the value at the point whose differential is to be found.
methods: 
1. evaluateDiff: Returns the differentiated value

integrate
Type: class
Takes a string of the function to be integrated and the limits over which the integration is to be performed.
methods: 
1. evaluateInt: Returns the integrated value

complex_number
params: Takes two floating point numbers real and imaginary parts of complex number.
Type: class
methods:
1. comp_scale: Takes a floating number as an argument. It scales the complex number by given value.
eg1. if z = complex_number(3,4); z.comp_scale(3) will return complex_number(9,12)

2. modulus: Takes no arguments. Returns the modulus  or length of the complex number. eg. As in eg1.  z.modulus() will return 5.0

3. argz: Takes no arguments. Returns the angle the complex number makes with the positive x-axis. eg. As in eg1 z.argz() will return 0.9272952060016121.

4. conjugate: Takes no arguments. Returns conjugate of the complex_number. eg. z.conjugate() will return complex_number(3,-4).

5. polar: Takes no arguments. Returns a tuple of the complex_number object in polar form. Eg. z.polar() returns (5.0,0.9272952060016121)

6. complex_power: Takes one args the argument with the power greater than or equal to one for which the complex_number is raised. Returns a complex_number. Eg. z.complex_power(1) returns complex_number(3,4)

7. inverse_complex_power: Takes one integer argument. The reciprocal of the number is the power to which the complex is raised. Returns a a list of complex_numbers of varying args but same modulus. 

8. nPolygon: Takes one argument of type integer. Draws a polygon of regular polygon of side n. Eg. z.nPolygon(3) will produce a triangle that is constrained by the circle of radius(r = 5.0) equal the modulus of the complex_number.

9. MandelBrotPlot: Takes one argument of type complex_number. Tries to plot the mandelbrot set.

complex_function
params: Takes two strings of function as argument.
Type: class
methods:
1. len(vectors[k])-iternum(vectors[k])-iternum complex_diff: Takes one floating point argument. That is the parameter value at the point the differential complexfunction is obtained. Returns a complex_number.

2. complex_int: Takes two floating point numbers as argument. The parameter number over which the itegration is performed. Returns a complex_number object. 

polynomial
params: Takes coefficient list of the polynomial vector. Eg. A polynomial like 2*x**2 + 3*x + 4 will be shown as [4,3,2]
Type: class
methods:
1. poly_eval: Takes one argument. Evaluates the polynomial at the given value.
2. 





