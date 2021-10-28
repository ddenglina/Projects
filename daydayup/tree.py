
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回对应节点TreeNode
    res = 0
    def KthNode(self, pRoot, k):  # 二叉搜索树的第K个节点
        # write code here
        if not pRoot or k == 0: return
        res = []

        def inorder(pRoot):
            if not pRoot or len(res) >= k:
                return
            inorder(pRoot.left)
            res.append(pRoot)
            inorder(pRoot.right)

        inorder(pRoot)
        if len(res) < k: return None
        return res[k - 1]


class Solution:
    # 树的子结构
    def HasSubtree(self, pRoot1: TreeNode, pRoot2: TreeNode) -> bool:
        # write code here
        if not pRoot1 or not pRoot2:
            return False
        if self.judge(pRoot1, pRoot2):
            return True
        return self.HasSubtree(pRoot1.left, pRoot2) or self.HasSubtree(pRoot1.right, pRoot2)

    def judge(self, tree, subtree):
        if not subtree: return True
        if not tree: return False
        if tree.val != subtree.val:
            return False
        return self.judge(tree.left, subtree.left) and self.judge(tree.right, subtree.right)

class Solution:
    # 二叉树的镜像
    def Mirror(self , pRoot ):
        # write code here
        if not pRoot:return
        if not pRoot.left and not pRoot.right:return pRoot
        pRoot.left,pRoot.right=pRoot.right,pRoot.left
        self.Mirror(pRoot.left)
        self.Mirror(pRoot.right)
        return pRoot


class Solution:
    # 二叉搜索树的后续遍历序列
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence: return False
        root = sequence[-1]

        for i in range(len(sequence)):
            if sequence[i] >= root:
                break
        for j in range(i, len(sequence)):
            if sequence[j] < root:
                return False

        left = True
        if i > 0:
            left = self.VerifySquenceOfBST(sequence[0:i])
        right = True
        if i < len(sequence) - 1:
            right = self.VerifySquenceOfBST(sequence[i:len(sequence) - 1])
        return left and right