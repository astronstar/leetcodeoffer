# 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

# 请写一个函数，求任意第n位对应的数字。

#  

# 示例 1：

# 输入：n = 3
# 输出：3
# 示例 2：

# 输入：n = 11
# 输出：0
class Solution:
    def findNthDigit(self, n: int) -> int:
        #1位数字1-9一共9个（1*9）
        #2位数字10-99一共90个 （2*90）
        #3位数字100-999一共900个（3*900）
        #4位1000-9999，9000个（4*9000）
        
        digit=1
        start=1
        count=9
        while n>count:
            n=n-count
            digit+=1
            start*=10
            count=digit*start*9
        return int(str(start+(n-1)//digit)[(n-1)%digit])