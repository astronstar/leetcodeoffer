#输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。
class Solution:
    def reverseWords(self, s: str) -> str:
        s=s.strip()
        s=s[::-1]
        res=""
        j=0
        for i in range(len(s)):
            if s[i]==' ' and s[i+1]!=' ':
                res+=s[j:i][::-1].strip()+' '
                j=i
        res+=s[j:][::-1].strip()
        return res