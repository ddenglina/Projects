# 链表环的入口节点
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # 链表中环的入口节点
        slow, fast = pHead, pHead
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast: break
        if not fast or not fast.next: # 无环，返回None
            return None
        slow = pHead
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return fast
# 两个链表的第一个公共节点
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if not pHead1 and not pHead2:return None
        if not pHead1 or not pHead2:return None
        p,q=pHead1,pHead2
        while p!=q:
            if not p:p=pHead2
            else:
                p=p.next
            if not q:q=pHead1
            else:
                q=q.next
        return p

# 复杂链表的复制
class Solution:
    # 删除链表中重复的节点
    def deleteDuplication(self, pHead):
        # write code here
        dummy = ListNode(0)
        cur = dummy
        while pHead:
            if pHead.next==None or pHead.next.val!=pHead.val:
                cur.next=pHead
                cur=pHead
            while pHead.next!=None and pHead.val == pHead.next.val:
                pHead=pHead.next
            pHead=pHead.next
        cur.next=None
        return dummy.next