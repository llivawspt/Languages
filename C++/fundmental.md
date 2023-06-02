# 1.引用
```c++
#include<iostream>
#inlcude<string>

using namespace std;

int main(void){
    int a = 0;
    int &num = a;
    //错误示范
    /*
    int &num;
    int &num = &a;
    */
    
    return 0;
}
```
注意事项：
+ 被引用的对象必须是一个对象，并且已经被初始化；
+ 在声明一个引用类型的变量的时候，必须进行初始化。
