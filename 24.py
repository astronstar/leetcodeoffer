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