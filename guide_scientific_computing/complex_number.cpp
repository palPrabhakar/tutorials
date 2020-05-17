#include <cmath>
#include "complex_number.hpp"

ComplexNumber::ComplexNumber() {
  mreal = 0.0;
  mimg  = 0.0;
}

ComplexNumber::ComplexNumber(double x, double y) {
  mreal = x;
  mimg = y;
}

double ComplexNumber::calculate_modulus() const {
  return sqrt(mreal*mreal + mimg*mimg);
}

double ComplexNumber::calculate_argument() const {
  return atan2(mimg, mreal);
}

ComplexNumber ComplexNumber::calculate_power(double n) const {
  double old_mod = calculate_modulus(); 
  double argument = calculate_argument();
  double new_mod = pow(old_mod, n);
  double x = new_mod*cos(n*argument);
  double y = new_mod*sin(n*argument);
  ComplexNumber new_complex(x, y);
  return new_complex;
}

ComplexNumber& ComplexNumber::operator=(const ComplexNumber& z) {
  mreal = z.mreal;
  mimg = z.mimg;
  return *this;
}

ComplexNumber ComplexNumber::operator-() const {
  ComplexNumber w(-mreal, -mimg);
  return w;
}

ComplexNumber ComplexNumber::operator+(const ComplexNumber& z) const {
  ComplexNumber w;
  w.mreal = mreal + z.mreal;
  w.mimg = mimg + z.mimg;
  return w;
}

ComplexNumber ComplexNumber::operator+(const ComplexNumber& z) const {
  ComplexNumber w;
  w.mreal = mreal - z.mreal;
  w.mimg = mimg - z.mimg;
  return w;
}
