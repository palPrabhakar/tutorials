#ifndef COMPLEX_H
#define COMPLEX_H

class ComplexNumber {
  private:
    double mreal;
    double mimg;
  public:
    //Constructors 
    ComplexNumber();
    ComplexNumber(double x, double y);

    //Method
    double calculate_modulus() const;
    double calculate_argument() const;
    ComplexNumber calculate_power(double n) const;

    //Helper Methods
    ComplexNumber& operator=(const ComplexNumber& z);
    ComplexNumber operator-() const;
    ComplexNumber operator+(const ComplexNumber& z) const;
    ComplexNumber operator-(const ComplexNumber& z) const;
    friend std::ostream& operator<<(std::ostream& output, const ComplexNumber& z);
};

#endif
