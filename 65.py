# 写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。
# 示例:

# 输入: a = 1, b = 1
# 输出: 2
class Solution:
    def add(self, a: int, b: int) -> int:
        trans=0xffffffff
        a,b=a&trans,b&trans
        while b!=0:
            carry=(a&b)<<1
            a^=b
            b=carry&trans
        if a<=0x7FFFFFFF:return a
        else:
            return ~(a^trans)