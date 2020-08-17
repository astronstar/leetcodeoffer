# 把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
# 你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
class Solution:
    def twoSum(self, n: int) -> List[float]:
        res=[]
        l1=[1/6]*6
        for i in range(1,n):
            tmp=[0]*(5*i+6)
            #l2=[1/6]*6
            for j in range(len(l1)):
                for k in range(6):
                    #print(len(tmp),j,k)
                    tmp[j+k]+=l1[j]*1/6
            l1=tmp
        return l1