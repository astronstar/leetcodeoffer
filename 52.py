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