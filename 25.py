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