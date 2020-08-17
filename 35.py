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