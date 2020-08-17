# 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

# 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

class Solution:
    def cuttingRope(self, n: int) -> int:
        if n<=3:
            return n-1
        div=n//3
        remain=n%3
        if remain==0:
            return 3**div%1000000007
        if remain==1:
            return 3**(div-1)*4%1000000007
        if remain==2:
            return 2*3**div%1000000007