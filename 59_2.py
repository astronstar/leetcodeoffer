# 请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

# 若队列为空，pop_front 和 max_value 需要返回 -1

# 示例 1：

# 输入: 
# ["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
# [[],[1],[2],[],[],[]]
# 输出: [null,null,null,2,1,2]
# 示例 2：

# 输入: 
# ["MaxQueue","pop_front","max_value"]
# [[],[],[]]
# 输出: [null,-1,-1]
class MaxQueue:

    def __init__(self):
        self.q=[]
        self.sort_q=[]


    def max_value(self) -> int:
        if self.q:
            return self.sort_q[0]
        else:
            return -1


    def push_back(self, value: int) -> None:
        self.q.append(value)
        
        while self.sort_q and self.sort_q[-1]<value:
            self.sort_q.pop(-1)
        self.sort_q.append(value)

    def pop_front(self) -> int:
        print(self.q,self.sort_q)
        if not self.q:
            return -1
        else:
            x=self.q.pop(0)
            if x==self.sort_q[0]:
                self.sort_q.pop(0)
        return x    