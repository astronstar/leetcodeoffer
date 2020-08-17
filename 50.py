# 在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

# 示例:
# s = "abaccdeff"
# 返回 "b"
# s = "" 
# 返回 " "
class Solution:
    def firstUniqChar(self, s: str) -> str:
        c=Counter(s)
        uniqset=set()
        for k,v in c.items():
            if v==1:
                uniqset.add(k)
        for i in s:
            if i in uniqset:
                return i
        return " "