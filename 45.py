# 数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。
# 请写一个函数，求任意第n位对应的数字。
# 示例 1：
# 输入：n = 3
# 输出：3
# 示例 2：
# 输入：n = 11
# 输出：0
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        def rules(a,b):
            if a+b>b+a:return 1
            if a+b<b+a:return -1
            else:return 0

        nums=[str(i) for i in nums]

        nums.sort(key=functools.cmp_to_key(rules))
        return "".join(nums)