# 输入一个字符串，打印出该字符串中字符的所有排列。
# 你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。
class Solution:
    def permutation(self, s: str) -> List[str]:
        s="".join(sorted(s))
        res=[]
        def dfs(s,tmp):
            if len(s)==0:
                res.append(tmp)
            for i in range(len(s)):
                if i-1>=0 and s[i]==s[i-1]:
                    continue
                else:
                    dfs(s[0:i]+s[i+1:],tmp+s[i])
        dfs(s,"")
        return res
 