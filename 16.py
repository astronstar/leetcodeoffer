# 实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
# 示例 1:

# 输入: 2.00000, 10
# 输出: 1024.00000
# 示例 2:

# 输入: 2.10000, 3
# 输出: 9.26100
# 示例 3:

# 输入: 2.00000, -2
# 输出: 0.25000
# 解释: 2-2 = 1/22 = 1/4 = 0.25
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:return 1
        elif n<0: return 1.0/self.myPow(x,-n)
        elif n%2==0:return self.myPow(x*x,n//2)
        else:return self.myPow(x*x,n//2)*x