#10_1.py
```python3
# 写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项。斐波那契数列的定义如下：

# F(0) = 0,   F(1) = 1
# F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
# 斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

# 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

#  

# 示例 1：

# 输入：n = 2
# 输出：1
# 示例 2：

# 输入：n = 5
# 输出：5
class Solution:
    def fib(self, n: int) -> int:
        if n==0:return 0
        if n==1:return 1
        dp=[0]*(n+1)
        dp[0]=0
        dp[1]=1
        for i in range(2,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        print(dp)
        return dp[-1]%(1000000007)  
```
#10_2.py
```python3
# 一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。
# 答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。
# 示例 1：

# 输入：n = 2
# 输出：2
# 示例 2：

# 输入：n = 7
# 输出：21
# 示例 3：

# 输入：n = 0
# 输出：1
class Solution:
    def numWays(self, n: int) -> int:
        if n==0:return 1
        if n==1:return 1
        if n==2:return 2
        dp=[0]*(n)
        dp[0]=1
        dp[1]=2
        for i in range(2,n):
            dp[i]=dp[i-1]+dp[i-2]
        print(dp)
        return dp[-1]%(1000000007)

```
#11.py
```python3
# 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

# 示例 1：

# 输入：[3,4,5,1,2]
# 输出：1
# 示例 2：
# 输入：[2,2,2,0,1]
# 输出：0
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        left=0
        right=len(numbers)-1
        while left<right:
            mid=(left+right)//2
            if numbers[mid]>numbers[right]:
                left=mid+1
            elif numbers[mid]<numbers[right]:
                right=mid
            else:right-=1
        return numbers[left]
```
#12.py
```python3
# 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一格开始，每一步可以在矩阵中向左、右、上、下移动一格。如果一条路径经过了矩阵的某一格，那么该路径不能再次进入该格子。例如，在下面的3×4的矩阵中包含一条字符串“bfce”的路径（路径中的字母用加粗标出）。

# [["a","b","c","e"],
# ["s","f","c","s"],
# ["a","d","e","e"]]

# 但矩阵中不包含字符串“abfb”的路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入这个格子。

#  

# 示例 1：

# 输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
# 输出：true
# 示例 2：

# 输入：board = [["a","b"],["c","d"]], word = "abcd"
# 输出：false
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        dx=[0,-1,0,1]
        dy=[-1,0,1,0]
        def dfs(x,y,k):
            if word[k]!=board[x][y]:
                return False
            if k==len(word)-1:return True
            t=board[x][y]
            board[x][y]="*"
            for i in range(4):
                if 0<=x+dx[i]<=len(board)-1 and 0<=y+dy[i]<=len(board[0])-1:
                    if dfs(x+dx[i],y+dy[i],k+1):
                        return True
            board[x][y]=t

        for i in range(len(board)):
            for j in range(len(board[0])):
                print(i,j)
                if dfs(i,j,0):
                    return True
            
        return False
```
#13.py
```python3
# 地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

# 示例 1：
# 输入：m = 2, n = 3, k = 1
# 输出：3
# 示例 2：
# 输入：m = 3, n = 1, k = 0
# 输出：1
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def dfs(i, j, si, sj):
            if i>=m or j>=n or si+sj>k or (i,j) in visited:return 0
            visited.add((i,j))
            return 1+dfs(i+1,j,si+1 if (i+1)%10!=0 else si-8,sj)+dfs(i,j+1,si,sj+1 if (j+1)%10!=0 else sj-8)
        
        visited=set()
        return dfs(0,0,0,0)
```
#14_1.py
```python3
# 给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

# 示例 1：

# 输入: 2
# 输出: 1
# 解释: 2 = 1 + 1, 1 × 1 = 1
# 示例 2:

# 输入: 10
# 输出: 36
# 解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n<=3:
            return n-1
        div=n//3
        remain=n%3
        if remain==0:
            return 3**div
        if remain==1:
            return 3**(div-1)*4
        if remain==2:
            return 2*3**div

```
#14_2.py
```python3
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
```
#15.py
```python3
# 请实现一个函数，输入一个整数，输出该数二进制表示中 1 的个数。例如，把 9 表示成二进制是 1001，有 2 位是 1。因此，如果输入 9，则该函数输出 2。

# 示例 1：

# 输入：00000000000000000000000000001011
# 输出：3
# 解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
# 示例 2：

# 输入：00000000000000000000000010000000
# 输出：1
# 解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
# 示例 3：

# 输入：11111111111111111111111111111101
# 输出：31
# 解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。

class Solution:
    def hammingWeight(self, n: int) -> int:
        mask=1
        count=0
        for i in range(32):
            if n&mask>0:
                count+=1
            mask=mask<<1
        return count
```
#16.py
```python3
# 实现函数double Power(double base, int exponent)，求base的exponent次方。不得使用库函数，同时不需要考虑大数问题。
# 示例 1:

# 输入: 2.00000, 10
# 输出: 1024.00000
# 示例 2:

# 输入: 2.10000, 3
# 输出: 9.26100
# 示例 3:

# 输入: 2.00000, -2
# 输出: 0.25000
# 解释: 2-2 = 1/22 = 1/4 = 0.25
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:return 1
        elif n<0: return 1.0/self.myPow(x,-n)
        elif n%2==0:return self.myPow(x*x,n//2)
        else:return self.myPow(x*x,n//2)*x
```
#17.py
```python3
# 输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

# 示例 1:

# 输入: n = 1
# 输出: [1,2,3,4,5,6,7,8,9]
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        largest=9
        for i in range(1,n):
            largest=largest*10+9
        return [i for i in range(1,largest+1)]
```
#18.py
```python3
# 给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

# 返回删除后的链表的头节点。

# 注意：此题对比原题有改动

# 示例 1:

# 输入: head = [4,5,1,9], val = 5
# 输出: [4,1,9]
# 解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
# 示例 2:

# 输入: head = [4,5,1,9], val = 1
# 输出: [4,5,9]
# 解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        dummy=ListNode(-1)
        dummy.next=head
        if head.val==val:return head.next
        while head and head.next:
            if head.next.val==val:
                head.next=head.next.next
            head=head.next
        return dummy.next
```
#19.py
```python3
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
```
#20.py
```python3
#请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"-1E-16"、"0123"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。
class Solution:
    def isNumber(self, s: str) -> bool:

        state = [
            {},
            # 状态1,初始状态(扫描通过的空格)
            {"blank": 1, "sign": 2, "digit": 3, ".": 4},
            # 状态2,发现符号位(后面跟数字或者小数点)
            {"digit": 3, ".": 4},
            # 状态3,数字(一直循环到非数字)
            {"digit": 3, ".": 5, "e": 6, "blank": 9},
            # 状态4,小数点(后面只有紧接数字)
            {"digit": 5},
            # 状态5,小数点之后(后面只能为数字,e,或者以空格结束)
            {"digit": 5, "e": 6, "blank": 9},
            # 状态6,发现e(后面只能符号位, 和数字)
            {"sign": 7, "digit": 8},
            # 状态7,e之后(只能为数字)
            {"digit": 8},
            # 状态8,e之后的数字后面(只能为数字或者以空格结束)
            {"digit": 8, "blank": 9},
            # 状态9, 终止状态 (如果发现非空,就失败)
            {"blank": 9}
        ]
        cur_state = 1
        for c in s:
            if c.isdigit():
                c = "digit"
            elif c in " ":
                c = "blank"
            elif c in "+-":
                c = "sign"

            if c not in state[cur_state]:
                return False
            cur_state=state[cur_state][c]

        if cur_state not in [3,5,8,9]:
            return False
        return True
```
#21.py
```python3
#输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:
        j=len(nums)-1
        i=0
        while i<j:
            if nums[i]%2==0:
                if nums[j]%2==0:
                    j=j-1
                    continue
                else:
                    nums[i],nums[j]=nums[j],nums[i]
            i=i+1
        return nums
 
```
#22.py
```python3
# 输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。
class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast=head
        slow=head
        for i in range(k):
            fast=fast.next
        
        while fast:
            slow=slow.next
            fast=fast.next

        return slow
```
#24.py
```python3
# 定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。
# 示例:
# 输入: 1->2->3->4->5->NULL
# 输出: 5->4->3->2->1->NULL

# 限制：
# 0 <= 节点个数 <= 5000
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if not head:return []
        p=head
        q=p.next
        while q:
            tmp=q.next
            q.next=p
            p=q
            q=tmp
        head.next=None
        return p
```
#25.py
```python3
# 输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

# 示例1：

# 输入：1->2->4, 1->3->4
# 输出：1->1->2->3->4->4
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        l=ListNode(-1)
        head=l
        while l1 and l2:
            if l1.val<=l2.val:
                l.next=l1
                l1=l1.next
            else:
                l.next=l2
                l2=l2.next
            l=l.next
        
        while l1:
            l.next=l1
            l1=l1.next
            l=l.next
        while l2:
            l.next=l2
            l2=l2.next
            l=l.next
        return head.next
```
#26.py
```python3
# 输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

# B是A的子结构， 即 A中有出现和B相同的结构和节点值。

# 例如:
# 给定的树 A:

#      3
#     / \
#    4   5
#   / \
#  1   2
# 给定的树 B：

#    4 
#   /
#  1
# 返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

# 示例 1：

# 输入：A = [1,2,3], B = [3,1]
# 输出：false
# 示例 2：

# 输入：A = [3,4,5,1,2], B = [4,1]
# 输出：true
class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def dfs(a,b):
            if not b:return True
            if not a or a.val!=b.val:return False
            return dfs(a.left,b.left) and dfs(a.right,b.right)

        if not A or not B:return False
        return dfs(A,B) or self.isSubStructure(A.left,B) or self.isSubStructure(A.right,B)
```
#27.py
```python3
# 请完成一个函数，输入一个二叉树，该函数输出它的镜像。

# 例如输入：

#      4
#    /   \
#   2     7
#  / \   / \
# 1   3 6   9
# 镜像输出：

#      4
#    /   \
#   7     2
#  / \   / \
# 9   6 3   1
class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        def dfs(root):
            if root is None:return []
            root.left,root.right=root.right,root.left
            if root.left:
                dfs(root.left)
            if root.right:
                dfs(root.right)
            return root
        return dfs(root)
```
#28.py
```python3
# 请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

# 例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# 但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

#     1
#    / \
#   2   2
#    \   \
#    3    3
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        def dfs(L,R):
            if not L and not R:return True
            if not L or not R or L.val!=R.val:return False
            return dfs(L.right,R.left) and dfs(L.left,R.right)

        if not root:return True
        else:
            return dfs(root.left,root.right)
```
#29.py
```python3
# 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

#  

# 示例 1：

# 输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
# 输出：[1,2,3,6,9,8,7,4,5]
# 示例 2：

# 输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# 输出：[1,2,3,4,8,12,11,10,9,5,6,7]
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []
        while matrix:
            res.extend(matrix.pop(0))
            matrix=list(zip(*matrix))[::-1]
        return res
```
#3.py
```python3
# jc3找出数组中重复的数字。
# 在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。
# 示例 1：
# 输入：
# [2, 3, 1, 0, 2, 5, 3]
# 输出：2 或 3  
# 限制：
# 2 <= n <= 100000
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        c=Counter(nums)
        for k,v in c.items():
            if v>1:
                return k

```
#30.py
```python3
# 定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。
# 示例:

# MinStack minStack = new MinStack();
# minStack.push(-2);
# minStack.push(0);
# minStack.push(-3);
# minStack.min();   --> 返回 -3.
# minStack.pop();
# minStack.top();      --> 返回 0.
# minStack.min();   --> 返回 -2.
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack=[]
        self.minstack=[]

    def push(self, x: int) -> None:
        self.stack.append(x)
        if not self.minstack or x<=self.minstack[-1]:
            self.minstack.append(x)

    def pop(self) -> None:
        x=self.stack.pop()
        if x==self.minstack[-1]:
            self.minstack.pop()     

    def top(self) -> int:
        return self.stack[-1]

    def min(self) -> int:
        return self.minstack[-1]
```
#31.py
```python3
# 输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

# 示例 1：

# 输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
# 输出：true
# 解释：我们可以按以下顺序执行：
# push(1), push(2), push(3), push(4), pop() -> 4,
# push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
# 示例 2：

# 输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
# 输出：false
# 解释：1 不能在 2 之前弹出。
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack=[]
        p=0
        for item in pushed:
            stack.insert(0,item)
            while stack and popped[p]==stack[0]:
                p+=1
                stack.pop(0)
        
        return not stack
```
#32.py
```python3
# 从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。
# 例如:
# 给定二叉树: [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回：

# [3,9,20,15,7]
class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        q=[]
        if root:
            q.append(root)
        res=[]
        while q:
            node=q.pop(0)
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        return res 

```
#32_2.py
```python3
# 从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。
# 例如:
# 给定二叉树: [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回其层次遍历结果：

# [
#   [3],
#   [9,20],
#   [15,7]
# ]

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        q=[]
        if root:
            q.append(root)
        res=[]
        while q:
            node=q.pop(0)
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

        return res 
```
#32_3.py
```python3
# 请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
# 例如:
# 给定二叉树: [3,9,20,null,null,15,7],

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回其层次遍历结果：

# [
#   [3],
#   [20,9],
#   [15,7]
# ]
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        q=[]
        if root:
            q.append([root])
        res=[]
        while q:
            nodelist=q.pop(0)
            
            level=[]
            t=[]
            for node in nodelist:
                t.append(node.val)
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            if level:
                q.append(level)
            res.append(t)
        for i in range(len(res)):
            if i%2==1:
                res[i]=res[i][::-1]    
        return res 
```
#33.py
```python3
# 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。
# 参考以下这颗二叉搜索树：

#      5
#     / \
#    2   6
#   / \
#  1   3
# 示例 1：

# 输入: [1,6,3,2,5]
# 输出: false
# 示例 2：

# 输入: [1,3,2,6,5]
# 输出: true

class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        if not postorder or len(postorder) == 0:return True
        rootval=postorder[-1]
        n=len(postorder)
        mid=0
        for i in range(n):
            if postorder[i]>rootval:
                break
        mid=i
        for i in range(mid,n-1):
            if postorder[i]<rootval:
                return False
        left=True
        if mid>0:
            left=self.verifyPostorder(postorder[0:mid])
        right=True
        if i<n-1:
            right=self.verifyPostorder(postorder[mid:-1])
        return  left and right
```
#34.py
```python3
# 输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。
# 示例:
# 给定如下二叉树，以及目标和 sum = 22，

#               5
#              / \
#             4   8
#            /   / \
#           11  13  4
#          /  \    / \
#         7    2  5   1
# 返回:

# [
#    [5,4,11,2],
#    [5,8,4,5]
# ]
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        if not root:return []
        res=[]
        def dfs(root,total,t):
            if not root.left and not root.right:
                print(t,sum,total)
                if total==sum:
                    res.append(t)
            #t=t+[root.val]
            if root.left:
                dfs(root.left,total+root.left.val,t+[root.left.val])   
            if root.right:
                dfs(root.right,total+root.right.val,t+[root.right.val]) 

        dfs(root,root.val,[root.val])
        return res
```
#35.py
```python3
#请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        def dfs(root):
            if not root:return
            if root in visited:
                return visited[root]
           
            copy=Node(root.val,None,None)
            visited[root]=copy
            copy.next=dfs(root.next)
            copy.random=dfs(root.random)

            return copy

        visited={}
        return dfs(head)
```
#36.py
```python3
#输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        def dfs(cur):
            if not cur:return
            dfs(cur.left)
            if self.pre:
                self.pre.right=cur
                cur.left=self.pre
            else:
                self.head=cur
            self.pre=cur
            dfs(cur.right)
            
        if not root:return
        self.pre=None
        dfs(root)
        self.head.left=self.pre
        self.pre.right=self.head
        return self.head
```
#37.py
```python3
#请实现两个函数，分别用来序列化和反序列化二叉树。
class Codec:

    def serialize(self, root):
        res=[]
        def dfs(root):
            if not root:
                res.append("#")
                return
            res.append(str(root.val))
            #if root.left:
            dfs(root.left)
            #if root.right:
            dfs(root.right)
        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        d=iter(data.split(","))
        def dfs():
            
            s=next(d)
            if s=="#":return
            node=TreeNode(int(s))
            node.left=dfs()
            node.right=dfs()
            return node
        return dfs()
```
#38.py
```python3
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
 
```
#39.py
```python3
# 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
# 你可以假设数组是非空的，并且给定的数组总是存在多数元素。
# 示例 1:
# 输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
# 输出: 2

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums)//2]
```
#4.py
```python3
# 在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
# 示例:

# 现有矩阵 matrix 如下：

# [
#   [1,   4,  7, 11, 15],
#   [2,   5,  8, 12, 19],
#   [3,   6,  9, 16, 22],
#   [10, 13, 14, 17, 24],
#   [18, 21, 23, 26, 30]
# ]
# 给定 target = 5，返回 true。
# 给定 target = 20，返回 false。

class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix)==0:return False
        i=0
        j=len(matrix[0])-1
        while i<len(matrix) and j>=0:
            if matrix[i][j]<target:
                i+=1
            elif matrix[i][j]>target:
                j-=1
            else:
                return True
        return False
```
#40.py
```python3
# 输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。
# 示例 1：
# 输入：arr = [3,2,1], k = 2
# 输出：[1,2] 或者 [2,1]
# 示例 2：
# 输入：arr = [0,1,2,1], k = 1
# 输出：[0]
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        q=[]
        heapq.heapify(arr)
        for i in range(k):
            q.append(heapq.heappop(arr))
        res3=q
        return res3
```
#41.py
```python3
# 输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

# 要求时间复杂度为O(n)。
# 示例1:

# 输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
# 输出: 6
# 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp=[0]*len(nums)
        dp[0]=nums[0]
        
        for i in range(1,len(nums)):
            dp[i]=max(dp[i-1]+nums[i],nums[i])
        return max(dp)

```
#42.py
```python3
# 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。
# 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。
# 示例 1：
# 输入：n = 12
# 输出：5
# 示例 2：
# 输入：n = 13
# 输出：6
class Solution:
    def countDigitOne(self, n: int) -> int:
        res=0
        digit=1
        high,cur,low=n//10,n%10,0
        while high!=0 or cur!=0:
            if cur == 0: res += high * digit
            elif cur == 1: res += high * digit + low + 1
            else: res += (high + 1) * digit
            low+=cur*digit
            cur=high%10
            high=high//10
            digit*=10
        return res
```
#43.py
```python3
# 输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

# 例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

#  

# 示例 1：

# 输入：n = 12
# 输出：5
# 示例 2：

# 输入：n = 13
# 输出：6
class Solution:
    def countDigitOne(self, n: int) -> int:
        res=0
        digit=1
        high,cur,low=n//10,n%10,0
        while high!=0 or cur!=0:
            if cur == 0: res += high * digit
            elif cur == 1: res += high * digit + low + 1
            else: res += (high + 1) * digit
            low+=cur*digit
            cur=high%10
            high=high//10
            digit*=10
        return res
```
#44.py
```python3
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
```
#45.py
```python3
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
```
#46.py
```python3
# 给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

# 示例 1:

# 输入: 12258
# 输出: 5
# 解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
class Solution:
    def translateNum(self, num: int) -> int:
        s=str(num)
        res=1
        tmp=1
        for i in range(2,len(s)+1):
            if "10" <= s[i - 2:i] <= "25":
                c=res+tmp
            else:
                c=res
            tmp=res
            res=c

        return res
```
#47.py
```python3
# 在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

# 示例 1:

# 输入: 
# [
#   [1,3,1],
#   [1,5,1],
#   [4,2,1]
# ]
# 输出: 12
# 解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        m=len(grid)
        n=len(grid[0])
        dp=[[0]*n for _ in range(m)]
        dp[0][0]=grid[0][0]
        for i in range(1,n):
            dp[0][i]=dp[0][i-1]+grid[0][i]
        for i in range(1,m):
            dp[i][0]=dp[i-1][0]+grid[i][0]
        for i in range(1,len(grid)):
            for j in range(1,len(grid[0])):
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])+grid[i][j]

        return dp[-1][-1]
```
#48.py
```python3
# 请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。
# 示例 1:
# 输入: "abcabcbb"
# 输出: 3 
# 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
# 示例 2:
# 输入: "bbbbb"
# 输出: 1
# 解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
# 示例 3:
# 输入: "pwwkew"
# 输出: 3
# 解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
#      请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        tmp=0
        res=0
        d={}
        for i in range(len(s)):
            j=d.get(s[i],-100)
            d[s[i]]=i
            if i-j<=tmp:
                tmp=i-j
            else:
                tmp=tmp+1
            res=max(res,tmp)
            print(res,tmp)
        return res
```
#49.py
```python3
# 我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。
# 示例:

# 输入: n = 10
# 输出: 12
# 解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

class Solution:
    def nthUglyNumber(self, n: int) -> int:
        heap=[1]
        heapq.heapify(heap)
        i=0
        res=[]
        while i<n:
            cur=heapq.heappop(heap)
            if cur not in res:
                i+=1
                res.append(cur)
                for j in [2,3,5]:
                    
                    x=j*cur
                    heapq.heappush(heap,x)
        return res[-1]
```
#5.py
```python3
# 请实现一个函数，把字符串 s 中的每个空格替换成"%20"。
# 示例 1：

# 输入：s = "We are happy."
# 输出："We%20are%20happy."


class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix)==0:return False
        i=0
        j=len(matrix[0])-1
        while i<len(matrix) and j>=0:
            if matrix[i][j]<target:
                i+=1
            elif matrix[i][j]>target:
                j-=1
            else:
                return True
        return False
```
#50.py
```python3
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
```
#51.py
```python3
# 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。
# 示例 1:

# 输入: [7,5,6,4]
# 输出: 5
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        self.cnt=0
        def merge(nums,start,mid,end):
            tmp=[]
            i=start
            j=mid+1
            while i<=mid and j<=end:
                if nums[i]<=nums[j]:
                    tmp.append(nums[i])
                    i+=1
                else:
                    self.cnt+=mid-i+1
                    tmp.append(nums[j])
                    j+=1
            
            while i<=mid:
                tmp.append(nums[i])
                i+=1
            while j<=end:
                tmp.append(nums[j])
                j+=1
            for i in range(len(tmp)):
                nums[start+i]=tmp[i]

        def mergeSort(nums,start,end):
            if start>=end:return
            mid=(start+end)//2
            mergeSort(nums,start,mid)
            mergeSort(nums,mid+1,end)
            merge(nums,start,mid,end)

        mergeSort(nums,0,len(nums)-1)
        return self.cnt
```
#52.py
```python3
#输入两个链表，找出它们的第一个公共节点。
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        p=headA
        q=headB

        while p!=q:
            if not p:
                p=headB
            else:
                p=p.next
            if not q:
                q=headA 
            else:
                q=q.next
        return q
```
#53.py
```python3
# 统计一个数字在排序数组中出现的次数。

#  

# 示例 1:

# 输入: nums = [5,7,7,8,8,10], target = 8
# 输出: 2
# 示例 2:

# 输入: nums = [5,7,7,8,8,10], target = 6
# 输出: 0
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # l,r=0,len(nums)-1
        # while l<r:
        #     mid=(l+r)//2
        #     if nums[mid]>target:
        #         r=mid
        #     elif nums[mid]<target:
        #         l=mid+1
        # res1=l

        # l,r=0,len(nums)-1
        # while l<r:
        #     mid=(l+r)//2
        #     if nums[mid]>target:
        #         r=mid
        #     elif nums[mid]<target:
        #         l=mid+1
        # res2=l
        # return res2-res1+1
        import bisect
        left=bisect.bisect_left(nums,target)
        right=bisect.bisect(nums,target)
        return right-left
```
#53_1.py
```python3
# 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

#  

# 示例 1:

# 输入: [0,1,3]
# 输出: 2
# 示例 2:

# 输入: [0,1,2,3,4,5,6,7,9]
# 输出: 8
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        left=0
        right=len(nums)-1
        
        while left<=right:
            mid=(left+right)//2
            
            if nums[mid]==mid:
                left=mid+1
            else:
                right=mid-1
        
        return left
```
#54.py
```python3
#给定一棵二叉搜索树，请找出其中第k大的节点。
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:return 0
        res=[]
        def dfs(root):
                if root.left:
                    dfs(root.left)
                res.append(root.val)
                if root.right:
                    dfs(root.right)
        if root:        
            dfs(root)
        return res[len(res)-k]
```
#55.py
```python3
# 输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

# 例如：

# 给定二叉树 [3,9,20,null,null,15,7]，

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回它的最大深度 3 。
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        
        if not root:return 0
        if not root.left and not root.right:return 1
        if root.left and not root.right:
            return 1+self.maxDepth(root.left)
        if root.right and not root.left:
            return 1+self.maxDepth(root.right)
        if root.left and root.right:
            return 1+max(self.maxDepth(root.left),self.maxDepth(root.right))
```
#55_2.py
```python3
# 输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

# 示例 1:

# 给定二叉树 [3,9,20,null,null,15,7]

#     3
#    / \
#   9  20
#     /  \
#    15   7
# 返回 true 。

# 示例 2:

# 给定二叉树 [1,2,2,3,3,null,null,4,4]

#        1
#       / \
#      2   2
#     / \
#    3   3
#   / \
#  4   4
# 返回 false 。
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        if not root:return True
        return abs(self.depth(root.left)-self.depth(root.right))<=1 and self.isBalanced(root.left) and self.isBalanced(root.right)

    def depth(self,root):
        if not root:return 0
        return 1+max(self.depth(root.left),self.depth(root.right))

```
#56.py
```python3
# 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
# 示例 1：
# 输入：nums = [4,1,4,6]
# 输出：[1,6] 或 [6,1]
# 示例 2：
# 输入：nums = [1,2,10,4,1,4,3,3]
# 输出：[2,10] 或 [10,2]
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        c=Counter(nums)
        res=[]
        for k,v in c.items():
            if v==1:
                res.append(k)
        return res
```
#56_2.py
```python3
# 一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。
# 示例 1：
# 输入：nums = [4,1,4,6]
# 输出：[1,6] 或 [6,1]
# 示例 2：
# 输入：nums = [1,2,10,4,1,4,3,3]
# 输出：[2,10] 或 [10,2]
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        c=Counter(nums)
        res=[]
        for k,v in c.items():
            if v==1:
                res.append(k)
        return res
```
#57.py
```python3
# 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
# 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
# 示例 1：
# 输入：target = 9
# 输出：[[2,3,4],[4,5]]
# 示例 2：
# 输入：target = 15
# 输出：[[1,2,3,4,5],[4,5,6],[7,8]]
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i=0
        j=len(nums)-1
        while i<j:
            if nums[i]+nums[j]>target:
                j-=1
            elif nums[i]+nums[j]<target:
                i+=1
            else:
                return [nums[i],nums[j]]

```
#57_2.py
```python3
# 输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。
# 序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。
# 示例 1：
# 输入：target = 9
# 输出：[[2,3,4],[4,5]]
# 示例 2：
# 输入：target = 15
# 输出：[[1,2,3,4,5],[4,5,6],[7,8]]

class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        nums=[i for i in range(1,target+1)]
        res=[]
        i=target//2
        start=2
        while i>=0 and start<=target//2+1:
            if sum(nums[i:i+start])==target:
                res.append(nums[i:i+start])
                i=i-1
                start+=1
            if sum(nums[i:i+start])>target:
                i=i-1
            if sum(nums[i:i+start])<target:
                start+=1
        return sorted(res)

```
#58.py
```python3
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
```
#58_2.py
```python3
# 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

# 示例 1：

# 输入: s = "abcdefg", k = 2
# 输出: "cdefgab"
# 示例 2：

# 输入: s = "lrloseumgh", k = 6
# 输出: "umghlrlose"
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:]+s[:n]
```
#59.py
```python3
# 给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

# 示例:

# 输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
# 输出: [3,3,5,5,6,7] 
# 解释: 

#   滑动窗口的位置                最大值
# ---------------               -----
# [1  3  -1] -3  5  3  6  7       3
#  1 [3  -1  -3] 5  3  6  7       3
#  1  3 [-1  -3  5] 3  6  7       5
#  1  3  -1 [-3  5  3] 6  7       5
#  1  3  -1  -3 [5  3  6] 7       6
#  1  3  -1  -3  5 [3  6  7]      7
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        q=[]
        res=[]
        for i in range(len(nums)):
            if i-k>=0 and nums[i-k]==q[0]:
                q.pop(0)

            while q and nums[i]>q[-1]:
                q.pop()
            q.append(nums[i])    

            print(q)
            if i>=k-1:
                res.append(q[0])
        return res
```
#59_2.py
```python3
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
```
#6.py
```python3
# 输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。
# 示例 1：
# 输入：head = [1,3,2]
# 输出：[2,3,1]
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # res=[]
        # while head:
        #     res.append(head.val)
        #     head=head.next
        # return res[::-1]
        #辅助栈
        res=[]
        stack=[]
        while head:
            stack.insert(0,head)
            head=head.next
        
        while stack:
            res.append(stack.pop(0).val)
        return res
```
#60.py
```python3
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
```
#61.py
```python3
# 从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。
# 示例 1:

# 输入: [1,2,3,4,5]
# 输出: True
#  

# 示例 2:

# 输入: [0,0,1,2,5]
# 输出: True
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        nums.sort()
        count=0
        # nums for i in nums
        for i in range(len(nums)-1):
            if nums[i]==0:
                count+=1
                continue
            elif nums[i]==nums[i+1]:
                return False
        if nums[-1]-nums[count]<5:
            return True
        else:    
            return False
```
#62.py
```python3
# 0,1,,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

# 例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。
# 示例 1：

# 输入: n = 5, m = 3
# 输出: 3
# 示例 2：

# 输入: n = 10, m = 17
# 输出: 2
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        q=[i for i in range(n)] 
        last=0
        while len(q)>1:
            index=(last+m-1)%len(q)
            q.pop(index)
            last=index
            
        return q[-1]
```
#63.py
```python3
# 假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？
# 示例 1:

# 输入: [7,1,5,3,6,4]
# 输出: 5
# 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
#      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
# 示例 2:

# 输入: [7,6,4,3,1]
# 输出: 0
# 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit,temp=0,0

        for i in range(len(prices)-1):
            temp=max(0,temp+prices[i+1]-prices[i])
            profit=max(profit,temp)
        return profit
```
#64.py
```python3
# 求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

# 示例 1：

# 输入: n = 3
# 输出: 6
# 示例 2：

# 输入: n = 9
# 输出: 45
class Solution:
    def sumNums(self, n: int) -> int:
        
        return n and n+self.sumNums(n-1)
```
#65.py
```python3
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
```
#66.py
```python3
# 给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B 中的元素 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。

#  

# 示例:

# 输入: [1,2,3,4,5]
# 输出: [120,60,40,30,24]
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        if not a:return []
        left=[1]
        right=[1]
        for i in range(1,len(a)):
            left.append(left[-1]*a[i-1])
            right.insert(0,right[0]*a[len(a)-i])
        #print(right)
        res=[]
        for i in range(len(left)):
            res.append(left[i]*right[i])
        return res
```
#67.py
```python3
# 写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。

# 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

# 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

# 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

# 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

# 在任何情况下，若函数不能进行有效的转换时，请返回 0。

# 说明：

# 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

# 示例 1:

# 输入: "42"
# 输出: 42
# 示例 2:

# 输入: "   -42"
# 输出: -42
# 解释: 第一个非空白字符为 '-', 它是一个负号。
#      我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
# 示例 3:

# 输入: "4193 with words"
# 输出: 4193
# 解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
# 示例 4:

# 输入: "words and 987"
# 输出: 0
# 解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
#      因此无法执行有效的转换。
# 示例 5:

# 输入: "-91283472332"
# 输出: -2147483648
# 解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
#      因此返回 INT_MIN (−231) 。
class Solution:
    def strToInt(self, str: str) -> int:
        
        s=str.strip()
        if len(s)==0:return 0
        print(s)
        nums=0
        flag=True
        if s[0]=="-" and len(s[1:])>0:
            flag=False
            s=s[1:]
            print(s)
        elif s[0]=="+" and len(s[1:])>0:
            s=s[1:]
        print(s)
        for i in range(len(s)):
            if 0<=ord(s[i])-ord('0')<=9:
                nums=10*nums+ord(s[i])-ord('0')
            else:
                break
        if not flag:
            if nums<2**31:
                return -nums
            else:
                return -2**31
        else:
            if nums<2**31:
                return nums           
            else:
                return 2**31-1
```
#68.py
```python3
# 给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

# 例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

# 示例 1:

# 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
# 输出: 6 
# 解释: 节点 2 和节点 8 的最近公共祖先是 6。
# 示例 2:

# 输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
# 输出: 2
# 解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if root.val-p.val <0 and root.val-q.val<0:
                root=root.right
            if root.val-p.val>0 and root.val-q.val>0:
                root=root.left
            else:break
        return root
```
#68_2.py
```python3
# 给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

# 百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

# 例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]

# 示例 1:

# 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
# 输出: 3
# 解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
# 示例 2:

# 输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
# 输出: 5
# 解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root==p or root==q:return root
        l=self.lowestCommonAncestor(root.left,p,q)
        r=self.lowestCommonAncestor(root.right,p,q)
        #if not l and not r:return
        if not l:return r
        if not r: return l
        if l and r:
            return root
```
#7.py
```python3
# 输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# 例如，给出

# 前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7]
# 返回如下的二叉树：

#     3
#    / \
#   9  20
#     /  \
#    15   7

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder)==0:return None
        rootval=preorder[0]
        index=inorder.index(rootval)
        left=inorder[:index]
        right=inorder[index+1:]
        t=TreeNode(-1)
        t.val=rootval
        t.left=self.buildTree(preorder[1:index+1],left)
        t.right=self.buildTree(preorder[index+1:],right)
        return t

```
#9.py
```python3
# 用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )
# 示例 1：

# 输入：
# ["CQueue","appendTail","deleteHead","deleteHead"]
# [[],[3],[],[]]
# 输出：[null,null,3,-1]
# 示例 2：

# 输入：
# ["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
# [[],[],[5],[2],[],[]]
# 输出：[null,-1,null,null,5,2]
class CQueue:

    def __init__(self):
        self.a=[]
        #b是辅助栈
        self.b=[]


    def appendTail(self, value: int) -> None:
        while self.a:
            self.b.insert(0,self.a.pop(0))
        self.a.insert(0,value)
        while len(self.b):
            self.a.insert(0,self.b.pop(0))
        #print(self.a)


    def deleteHead(self) -> int:
        #print(self.a)
        if not self.a:return -1
        return self.a.pop(0)
```
#mkfile.py
```python3
import glob

res=[]
for filename in glob.glob('*.py'):
    f = open(filename,'r',encoding="utf-8")
    res.append("#"+filename+"\n")
    res.append("```python3\n")
    res.extend(f.readlines())
    res.append("\n")
    res.append("```\n")
    f.close()
print(len(res))
print(res[0])
f=open("all.py","w",encoding="utf-8")
for line in res:
    f.write(line)
f.close()
```
