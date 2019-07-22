//Exercise 3
// my first program in C++
#include <iostream>
#include <cmath>
using namespace std;
int main()
{
    float a, b, c,root1, root2 ;
    cout << "enter a, b, c";
    cin >> a >> b >> c;
    root1=(-b+sqrt(b*b-4*a*c))/(2*a);
    root2=(-b-sqrt(b*b-4*a*c))/(2*a);
    cout << "root1=" << root1 << "root2="<< root2 ;
    return 0;
}
