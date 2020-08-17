#请实现一个函数用来匹配包含'. '和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        slen=len(s)
        plen=len(p)
        dp=[[False]*(plen+1) for _ in  range(slen+1)]
        dp[0][0]=True
        for i in range(plen):
            if p[i]=="*":
                dp[0][i+1]=dp[0][i-1]
        
        for i in range(slen):
            for j in range(plen):
                if p[j]=="." or s[i]==p[j]:
                    dp[i+1][j+1]=dp[i][j]
                elif p[j]=="*":
                    if s[i]!=p[j-1]:
                        dp[i+1][j+1]=dp[i+1][j-1]
                    if p[j-1]=="." or s[i]==p[j-1]:
                        dp[i+1][j+1]=(dp[i][j+1] | dp[i+1][j] | dp[i+1][j-1])
                    
        print(dp)
        return dp[-1][-1]